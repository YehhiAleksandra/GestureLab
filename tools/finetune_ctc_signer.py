"""Дообучение CTC на жестах того же исполнителя (словарь Rutube-канала 23691449).

Зачем: фразовый замер этапа 14 дал 0% смысла, а на 893 изолированных клипах того же
канала 90% пустых глоссов и 1% попаданий — модель Slovo не переносится на этого
исполнителя. Здесь энкодер дообучается на клипах словаря, чьи слова уже есть среди 1000
классов модели (иначе голову пришлось бы расширять), Slovo повторяется, чтобы не
разучиться остальному. 67 фразовых клипов в обучение не входят — это тест.

Протокол: фиксированное число эпох, итоговый (не лучший на отложенных) checkpoint,
отложенная доля классов словаря, контроль на тестовых потоках Slovo.

  python tools/finetune_ctc_signer.py --epochs 25 --steps 120
  python tools/finetune_ctc_signer.py --dev-frac 0 --out models/slovo_ctc_signer_all.pt
  python tools/finetune_ctc_signer.py --selftest
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from gloss_coverage import norm, split_classes  # noqa: E402
from sign_stream import CTCSignNet, greedy_decode, load_ctc, out_lengths  # noqa: E402
from slovo_model import augment, resample  # noqa: E402
from train_ctc_slovo import (  # noqa: E402
    batch_features, corpus_phrases, edit_counts, evaluate, evaluate_corpus,
    file_sha256, git_commit, make_corpus_stream, make_stream, prepare_slovo,
    spec_augment, warm_start)


def class_index(names):
    """Глосса → индекс класса: точное имя и синонимы через «/» (как в Easy_sign)."""
    out = {}
    for i, n in enumerate(names):
        for part in split_classes(n):
            out.setdefault(part, i)
    return out


def load_signer(path, names, dev_frac=0.2, seed=0):
    """Клипы словаря → [(класс, кадры)]; отложенная доля считается по классам.

    Слово с несколькими клипами отдаёт один клип в отложенные, слово с единственным
    клипом уходит в отложенные целиком с вероятностью dev_frac. Так отложенные —
    это слова, которых модель при дообучении не видела вовсе.
    """
    with np.load(path) as d:
        frames, lens, words = d['frames'], d['lens'].astype(int), [str(w) for w in d['words']]
    offs = np.concatenate([[0], np.cumsum(lens)])
    index = class_index(names)
    by_class, skipped = {}, 0
    for k, w in enumerate(words):
        c = index.get(norm(w))
        if c is None:
            skipped += 1
            continue
        by_class.setdefault(c, []).append(frames[offs[k]:offs[k + 1]])

    rng = np.random.default_rng(seed)
    train, dev = [], []
    for c in sorted(by_class):
        pool = by_class[c]
        if len(pool) >= 2 and dev_frac > 0:
            order = rng.permutation(len(pool))
            dev.append((c, pool[int(order[0])]))
            train += [(c, pool[int(i)]) for i in order[1:]]
        elif dev_frac > 0 and rng.random() < dev_frac:
            dev.append((c, pool[0]))
        else:
            train += [(c, s) for s in pool]
    return {'train': train, 'dev': dev, 'skipped': skipped, 'classes': len(by_class)}


def mix_item(seqs_by_class, classes, corpus_train, signer_train, rng,
             corpus_frac=0.45, signer_frac=0.35, single_frac=0.05):
    """Один CTC-пример: корпус Slovo, одиночный жест исполнителя, одиночный Slovo, поток."""
    r = rng.random()
    if corpus_train and r < corpus_frac:
        return make_corpus_stream(seqs_by_class, corpus_train, rng)
    if signer_train and r < corpus_frac + signer_frac:
        c, s = signer_train[int(rng.integers(len(signer_train)))]
        return resample(augment(s, rng), int(rng.integers(16, 49))), [c]
    if r < corpus_frac + signer_frac + single_frac:
        c = int(rng.choice(classes))
        pool = seqs_by_class[c]
        s = augment(pool[int(rng.integers(len(pool)))], rng)
        return resample(s, int(rng.integers(16, 49))), [c]
    return make_stream(seqs_by_class, classes, rng)


def mix_aux(seqs_by_class, classes, signer_train, rng, n, feat, signer_frac=0.5):
    """Изолированные клипы для вспомогательной CE: половина — жесты исполнителя."""
    streams, labels = [], []
    for _ in range(n):
        if signer_train and rng.random() < signer_frac:
            c, s = signer_train[int(rng.integers(len(signer_train)))]
            s = augment(s, rng)
        else:
            c = int(rng.choice(classes))
            pool = seqs_by_class[c]
            s = augment(pool[int(rng.integers(len(pool)))], rng)
        streams.append(resample(s, int(rng.integers(16, 49))))
        labels.append(c)
    x, lens = batch_features(streams, feat)
    return x, lens, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def signer_metrics(model, clips, dev='cpu'):
    """Greedy по одному клипу: попадание класса (в любом месте выхода) и доля пустых."""
    model.eval()
    hit = single = empty = 0
    for c, s in clips:
        x, lens = batch_features([s], model.inp[0].in_features)
        pred, _ = greedy_decode(model(x.to(dev), lens.to(dev))[0].cpu())
        hit += int(c in pred)
        single += int(pred == [c])
        empty += int(not pred)
    n = max(len(clips), 1)
    return {'clips': len(clips), 'hit': round(hit / n, 3), 'single': round(single / n, 3),
            'empty': round(empty / n, 3)}


def run(a):
    if Path(a.out).exists():
        sys.exit(f'{a.out} уже существует — выберите другое имя')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    names = list(ck['names'])
    feat_in = ck['feat']
    arch = ck['arch']

    d = prepare_slovo(a.data)
    seqs_by_class, classes, te_by_class, te_classes = (
        d['seqs_by_class'], d['classes'], d['te_by_class'], d['te_classes'])
    sig = load_signer(a.dict_data, names, a.dev_frac, a.seed)
    print(f'Slovo: классов {len(names)}, train клипов {d["n_train"]}, test {d["n_test"]}; '
          f'словарь исполнителя: {len(sig["train"])} train, {len(sig["dev"])} отложено, '
          f'не совпало с классами {sig["skipped"]}', flush=True)

    index = {n.lower(): i for i, n in enumerate(names)}
    corpus_train, corpus_test = [], []
    if a.corpus_frac > 0:
        try:
            corpus_train = corpus_phrases(index, 'train')
            corpus_test = [p for p in corpus_phrases(index, 'test')
                           if all(c in te_by_class for c in p)]
        except Exception as e:  # noqa: BLE001 — внешний сервис: любая ошибка → без корпуса
            print(f'корпус недоступен ({e}) — без потоков корпуса', flush=True)
    print(f'корпус: train {len(corpus_train)} фраз, test {len(corpus_test)}', flush=True)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    aux_n = len(names) if a.aux_weight > 0 else 0
    model = CTCSignNet(len(names), d=arch['d'], layers=arch['layers'],
                       dropout=arch['dropout'], feat=feat_in, aux=aux_n).to(dev)
    warm_start(model, a.init, names)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.steps,
                                                pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    t0 = time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(a.steps):
            streams, labels = [], []
            for _ in range(a.bs):
                s, lab = mix_item(seqs_by_class, classes, corpus_train, sig['train'], rng,
                                  a.corpus_frac, a.signer_frac, a.single_frac)
                streams.append(s)
                labels.append(lab)
            x, lens = batch_features(streams, feat_in)
            if a.specaug:
                x = spec_augment(x, lens.numpy(), rng)
            targets = torch.tensor([c + 1 for lab in labels for c in lab], dtype=torch.long)
            tlens = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
            logits = model(x.to(dev), lens.to(dev))
            loss = ctc(logits.log_softmax(-1).transpose(0, 1), targets.to(dev),
                       torch.from_numpy(out_lengths(lens.numpy())).to(dev), tlens.to(dev))
            if aux_n:
                ax, alens, ay = mix_aux(seqs_by_class, classes, sig['train'], rng, a.aux_bs,
                                        feat_in)
                if a.specaug:
                    ax = spec_augment(ax, alens.numpy(), rng)
                loss = loss + a.aux_weight * F.cross_entropy(
                    model(ax.to(dev), alens.to(dev), aux=True)[1], ay.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss)

        if ep % a.eval_every == 0 or ep == a.epochs:
            ev = evaluate(model, te_by_class, te_classes, np.random.default_rng(1234),
                          n=60, dev=dev)
            line = (f'ep {ep:3d} loss {total / a.steps:.3f} | slovo gloss {ev["gloss_acc"]:.3f}'
                    f' edit {ev["edit_rate"]:.3f}')
            if sig['dev']:
                dv = signer_metrics(model, sig['dev'], dev)
                tr = signer_metrics(model, sig['train'][:120], dev)
                line += (f' | словарь отложено hit {dv["hit"]:.3f} single {dv["single"]:.3f}'
                         f' пусто {dv["empty"]:.3f} | train hit {tr["hit"]:.3f}')
            print(line + f' [{time.time() - t0:.0f} с]', flush=True)

    torch.save({'model': model.state_dict(), 'names': names, 'feat': feat_in, 'arch': arch,
                'aux': aux_n, 'kind': 'ctc_holistic',
                'config': {'data': a.data, 'dict_data': a.dict_data, 'init': a.init,
                           'epochs': a.epochs, 'steps': a.steps, 'bs': a.bs, 'lr': a.lr,
                           'seed': a.seed, 'corpus_frac': a.corpus_frac,
                           'signer_frac': a.signer_frac, 'single_frac': a.single_frac,
                           'aux_weight': a.aux_weight, 'dev_frac': a.dev_frac,
                           'n_signer_train': len(sig['train']),
                           'n_signer_dev': len(sig['dev']), 'motion': True}},
               a.out)

    model2, names2, _ = load_ctc(a.out, dev)
    assert names2 == names
    res = {'checkpoint': Path(a.out).name, 'checkpoint_sha256': file_sha256(a.out),
           'slovo_clean': evaluate(model2, te_by_class, te_classes,
                                   np.random.default_rng(1234), n=60, dev=dev),
           'signer_dev': signer_metrics(model2, sig['dev'], dev) if sig['dev'] else None,
           'signer_train': signer_metrics(model2, sig['train'], dev)}
    if corpus_test:
        res['slovo_corpus'] = evaluate_corpus(model2, te_by_class, corpus_test,
                                              np.random.default_rng(1234), n=60, dev=dev)
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    Path('reports/evidence').mkdir(parents=True, exist_ok=True)
    Path(f'reports/evidence/{Path(a.out).stem}_eval.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    manifest = {'schema_version': 1, 'kind': 'ctc_holistic',
                'dataset': a.data, 'dataset_sha256': file_sha256(a.data),
                'dict_dataset': a.dict_data, 'dict_dataset_sha256': file_sha256(a.dict_data),
                'checkpoint_sha256': res['checkpoint_sha256'],
                'init_sha256': file_sha256(a.init),
                'config': {'epochs': a.epochs, 'steps': a.steps, 'bs': a.bs, 'lr': a.lr,
                           'seed': a.seed, 'corpus_frac': a.corpus_frac,
                           'signer_frac': a.signer_frac, 'single_frac': a.single_frac,
                           'aux_weight': a.aux_weight, 'dev_frac': a.dev_frac},
                'arch': arch, 'git_commit': git_commit(),
                'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    Path(str(a.out) + '.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'сохранено {a.out} и манифест', flush=True)


def selftest():
    names = ['мать', 'там/туда', 'улица']
    assert class_index(names) == {'мать': 0, 'там': 1, 'туда': 1, 'улица': 2}
    frames = np.zeros((10, 59, 3), np.float32)
    words = np.array(['Мать', 'туда', 'молоко', 'Улица'])
    lens = np.array([3, 4, 2, 1])
    path = Path('reports/evidence/_signer_selftest.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=frames, lens=lens, words=words)
    try:
        d = load_signer(path, names, dev_frac=1.0, seed=0)
        assert len(d['train']) == 0 and len(d['dev']) == 3, d
        assert d['skipped'] == 1  # «молоко» нет среди классов
        assert [c for c, _ in d['dev']] == [0, 1, 2]
        d = load_signer(path, names, dev_frac=0.0, seed=0)
        assert len(d['train']) == 3 and not d['dev']
        assert all(len(s) and s.shape[1] == 59 for _, s in d['train'])
        # у класса два клипа: при dev_frac=0 оба обязаны остаться в обучении
        np.savez_compressed(path, frames=frames, lens=lens[1:3], words=np.array(['мать', 'мать']))
        d = load_signer(path, names, dev_frac=0.0, seed=0)
        assert len(d['train']) == 2 and not d['dev'], d
    finally:
        path.unlink()
    rng = np.random.default_rng(0)
    by_class = {0: [frames], 1: [frames]}
    stream, labels = mix_item(by_class, [0, 1], [], [(1, frames)], rng,
                              corpus_frac=0.0, signer_frac=1.0, single_frac=0.0)
    assert labels == [1] and stream.shape[1:] == (59, 3)
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/slovo/slovo_holistic.npz')
    ap.add_argument('--dict-data', default='datasets/youtube_dict/rutube_partial.npz')
    ap.add_argument('--init', default='models/slovo_ctc_motion.pt')
    ap.add_argument('--out', default='models/slovo_ctc_motion_signer.pt')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--corpus-frac', type=float, default=0.45)
    ap.add_argument('--signer-frac', type=float, default=0.35)
    ap.add_argument('--single-frac', type=float, default=0.05)
    ap.add_argument('--aux-weight', type=float, default=0.3)
    ap.add_argument('--aux-bs', type=int, default=16)
    ap.add_argument('--dev-frac', type=float, default=0.2,
                    help='доля классов словаря в отложенные; 0 — учить на всех клипах')
    ap.add_argument('--specaug', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)


if __name__ == '__main__':
    main()
