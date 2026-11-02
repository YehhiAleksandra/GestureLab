"""Дообучение CTC на словаре канала: голова расширяется на все 878 слов, энкодер
адаптируется к исполнителю на 893 его клипах — без камеры.

Зачем: 690 клипов словаря не выражаются классами головы (этап 15), а 120 из 146 слов
фраз существуют в одной записи (этапы 19–20). Здесь все слова словаря становятся
классами, и CTC учится выдавать их цепочки на синтетике из клипов того же
исполнителя: recombine 2–5 клипов словаря + корпус РЖЯ + фразовые клипы обучающей
части фолда + одиночные Slovo. Фолд-протокол честен только относительно данных,
исключённых из обучения: --init обязан быть чекпойнтом, который не видел фразовых
клипов (дефолт phrase20/prod/prod.pt обучен на всех 67 — B62, см. отчёт 27 §15).

  python tools/finetune_dict.py --selftest
  python tools/finetune_dict.py                     # 5 фолдов, ~8 мин на GPU
  python tools/finetune_dict.py --train-all         # + production-чекпойнт
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from eval_phrase_clips import words  # noqa: E402
from expand_ctc_head import expand_checkpoint  # noqa: E402
from finetune_ctc_phrases import canon_map, folds, load_phrase_clips  # noqa: E402
from centroid_eval import load_dict_clips  # noqa: E402
from finetune_ctc_signer import class_index  # noqa: E402
from gloss_coverage import norm  # noqa: E402
from sign_stream import CTCSignNet  # noqa: E402
from slovo_model import augment, resample  # noqa: E402
from train_ctc_slovo import (batch_features, corpus_phrases, evaluate,  # noqa: E402
                             file_sha256, git_commit, make_corpus_stream,
                             make_stream, prepare_slovo, spec_augment, warm_start)


def mix_dict(dict_pool, dict_classes, seqs_by_class, classes, corpus, phrases, rng,
             dict_frac=0.30, corpus_frac=0.20, phrase_frac=0.25, single_frac=0.15):
    """Один CTC-пример: цепочка клипов словаря, корпус, фраза фолда, одиночный,
    поток Slovo. Порядок как в mix_phrase (этап 19), источник слов — словарь."""
    r = rng.random()
    if dict_pool and r < dict_frac:
        return recombine_dict(dict_pool, dict_classes, rng)
    if corpus and r < dict_frac + corpus_frac:
        return make_corpus_stream(seqs_by_class, corpus, rng)
    if phrases and r < dict_frac + corpus_frac + phrase_frac:
        s, labels = phrases[int(rng.integers(len(phrases)))]
        return augment(s, rng), list(labels)
    if dict_pool and r < dict_frac + corpus_frac + phrase_frac + single_frac:
        c = int(rng.choice(dict_classes))
        pool = dict_pool[c]
        s = augment(pool[int(rng.integers(len(pool)))], rng)
        return resample(s, int(rng.integers(16, 49))), [c]
    return make_stream(seqs_by_class, classes, rng)


def recombine_dict(pool, classes, rng, kmin=2, kmax=5):
    """Синтетическая цепочка из клипов словаря: 2–5 разных слов с паузами 2–4 кадра."""
    hi = min(kmax, len(classes))
    k = int(rng.integers(min(kmin, hi), hi + 1))
    labels = [int(c) for c in rng.choice(classes, k, replace=False)]
    parts, prev = [], None
    for c in labels:
        seg = pool[c][int(rng.integers(len(pool[c])))]
        seg = seg[1:-1] if len(seg) > 6 else seg
        if prev is not None:
            parts.append(np.repeat(prev[-1:], int(rng.integers(2, 5)), axis=0))
        parts.append(resample(augment(seg, rng), int(rng.integers(16, 49))))
        prev = parts[-1]
    return np.concatenate(parts).astype(np.float32), labels


def build_dict_pool(dict_words, names):
    """[(слово, кадры)] → (pool{класс: [кадры]}, классы, покрыто клипов)."""
    index = class_index(names)
    pool, covered = {}, 0
    for w, frames in dict_words:
        c = index.get(norm(w))
        if c is None:
            continue
        pool.setdefault(c, []).append(frames)
        covered += 1
    return pool, sorted(pool), covered


def load_extra_dict(path, cap=8):
    """Дополнительный словарь (npz с lens или offsets) → [(слово, кадры)].

    Метки чистятся: «- Абрикос» → «абрикос»; мусор фильтруется (короче 3 символов,
    цифры как в «вариант 1»); на слово остаётся не более cap клипов — повторы
    полезны, но доминирование одного слова в батче вредно.
    """
    with np.load(path) as d:
        frames = d['frames']
        if 'lens' in d:
            lens = d['lens'].astype(int)
        else:
            lens = np.diff(d['offsets'].astype(int))
        ws = [str(x) for x in d['words']]
    offs = np.concatenate([[0], np.cumsum(lens)])
    out, per_word = [], {}
    for k, w in enumerate(ws):
        w = norm(re.sub(r'^[-\s]+', '', w))
        if len(w) < 3 or any(ch.isdigit() for ch in w):
            continue
        if per_word.get(w, 0) >= cap:
            continue
        per_word[w] = per_word.get(w, 0) + 1
        out.append((w, frames[offs[k]:offs[k + 1]].astype(np.float32)))
    return out


def train_fold(names, init_path, arch, feat, train_clips, dict_pool, dict_classes,
               seqs_by_class, classes, corpus, a, fold, dev):
    """Один фолд: тёплый старт от расширенной головы, смесь mix_dict, CTC."""
    torch.manual_seed(a.seed + fold)
    rng = np.random.default_rng(a.seed + fold)
    model = CTCSignNet(len(names), d=arch['d'], layers=arch['layers'],
                       dropout=arch['dropout'], feat=feat, aux=0).to(dev)
    warm_start(model, init_path, names)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr,
                                                total_steps=a.epochs * a.steps,
                                                pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    phrases = [(c['span'], c['labels']) for c in train_clips]
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(a.steps):
            streams, labels = [], []
            for _ in range(a.bs):
                s, lab = mix_dict(dict_pool, dict_classes, seqs_by_class, classes,
                                  corpus, phrases, rng, a.dict_frac, a.corpus_frac,
                                  a.phrase_frac, a.single_frac)
                streams.append(s)
                labels.append(lab)
            x, lens = batch_features(streams, feat)
            if a.specaug:
                x = spec_augment(x, lens.numpy(), rng)
            targets = torch.tensor([c + 1 for lab in labels for c in lab],
                                   dtype=torch.long)
            tlens = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
            logits = model(x.to(dev), lens.to(dev))
            loss = ctc(logits.log_softmax(-1).transpose(0, 1), targets.to(dev),
                       torch.from_numpy((lens.numpy() + 1) // 2).to(dev), tlens.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss)
        if ep % 5 == 0 or ep == a.epochs:
            print(f'  фолд {fold} ep {ep:3d} loss {total / a.steps:.3f}', flush=True)
    return model


def run(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    dict_words = load_dict_clips(a.dict_data)
    for extra in [x for x in a.extra_dict.split(',') if x]:
        ex = load_extra_dict(extra)
        print(f'доп. словарь {Path(extra).name}: {len(ex)} клипов', flush=True)
        dict_words += ex
    expanded = expand_checkpoint(ck, [w for w, _ in dict_words])
    names, arch, feat = list(expanded['names']), ck['arch'], ck['feat']
    init_path = work / 'init_dict_expanded.pt'
    torch.save(expanded, init_path)
    print(f"классы {len(ck['names'])} → {len(names)} "
          f"(+{expanded['expanded_added']})", flush=True)

    dict_pool, dict_classes, covered = build_dict_pool(dict_words, names)
    print(f'словарь: клипов покрыто {covered}/{len(dict_words)}, '
          f'классов {len(dict_classes)}', flush=True)

    inv = json.loads(Path(a.inventory).read_text(encoding='utf-8'))
    plain = [w for p in inv['phrases'] if p.get('duration') for w in words(p['phrase'])]
    canon = canon_map(plain, {n.lower() for n in names}) if a.canon else {}
    clips, skipped = load_phrase_clips(a.cache, a.inventory, names, canon)
    print(f'фразовых клипов {len(clips)}, пропущено {len(skipped)}; '
          f'канонизовано {sum(1 for w in plain if canon.get(w, w) != w)}', flush=True)

    d = prepare_slovo(a.data)
    seqs_by_class, classes = d['seqs_by_class'], d['classes']
    index = {n.lower(): i for i, n in enumerate(names)}
    try:
        # только фразы, целиком покрытые классами Slovo: make_corpus_stream
        # берёт клипы из seqs_by_class, расширенные id словаря там отсутствуют
        corpus = [p for p in corpus_phrases(index, 'train')
                  if all(i in seqs_by_class for i in p)]
    except Exception as e:  # noqa: BLE001 — без корпуса остаются словарь и Slovo
        print(f'корпус недоступен ({e})', flush=True)
        corpus = []
    print(f'Slovo train {d["n_train"]}, корпус {len(corpus)} фраз', flush=True)

    parts = folds(len(clips), a.folds, a.seed)
    fold_report = []
    for k, held_idx in enumerate(parts):
        train = [clips[i] for i, c in enumerate(clips) if i not in set(held_idx)]
        model = train_fold(names, str(init_path), arch, feat, train, dict_pool,
                           dict_classes, seqs_by_class, classes, corpus, a, k, dev)
        path = work / f'fold{k}.pt'
        torch.save({'model': model.state_dict(), 'names': names, 'feat': feat,
                    'arch': arch, 'aux': 0, 'kind': 'ctc_holistic',
                    'config': {'init': a.init, 'epochs': a.epochs, 'steps': a.steps,
                               'bs': a.bs, 'lr': a.lr, 'seed': a.seed + k,
                               'dict_clips': covered, 'train_clips': len(train),
                               'held_clips': len(held_idx)}}, path)
        slovo = evaluate(model, d['te_by_class'], d['te_classes'],
                         np.random.default_rng(1234), n=30, dev=dev)
        fold_report.append({'fold': k, 'pt': str(path),
                            'sha256': file_sha256(path),
                            'slovo_clean_gloss': round(slovo['gloss_acc'], 3)})
        print(f'фолд {k} сохранён, slovo clean {slovo["gloss_acc"]:.3f} '
              f'[{time.time() - t0:.0f} с]', flush=True)

    if a.train_all:
        model = train_fold(names, str(init_path), arch, feat, clips, dict_pool,
                           dict_classes, seqs_by_class, classes, corpus, a,
                           a.folds, dev)
        prod = work / 'prod_dict.pt'
        torch.save({'model': model.state_dict(), 'names': names, 'feat': feat,
                    'arch': arch, 'aux': 0, 'kind': 'ctc_holistic',
                    'config': {'init': a.init, 'epochs': a.epochs, 'steps': a.steps,
                               'bs': a.bs, 'lr': a.lr, 'seed': a.seed,
                               'dict_clips': covered, 'train_clips': len(clips),
                               'held_clips': 0}}, prod)
        slovo = evaluate(model, d['te_by_class'], d['te_classes'],
                         np.random.default_rng(1234), n=200, dev=dev)
        print(f'prod_dict: slovo clean {slovo["gloss_acc"]:.3f} (n=200)', flush=True)

    result = {'tool': 'finetune_dict', 'date': time.strftime('%Y-%m-%d'),
              'init': Path(a.init).name, 'init_sha256': file_sha256(a.init),
              'classes': len(names), 'expanded_added': expanded['expanded_added'],
              'dict_clips': covered, 'clips': len(clips), 'skipped': skipped,
              'folds': a.folds, 'seed': a.seed,
              'seconds': round(time.time() - t0, 1), 'folds_report': fold_report,
              'config': {'epochs': a.epochs, 'steps': a.steps, 'bs': a.bs, 'lr': a.lr,
                         'dict_frac': a.dict_frac, 'corpus_frac': a.corpus_frac,
                         'phrase_frac': a.phrase_frac, 'single_frac': a.single_frac,
                         'specaug': a.specaug}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    manifest = {'schema_version': 1, 'kind': 'ctc_holistic_dict_folds',
                'init': a.init, 'init_sha256': file_sha256(a.init),
                'classes': len(names), 'dict_clips': covered, 'git_commit': git_commit(),
                'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    Path(a.out).with_suffix('.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'→ {a.out}', flush=True)


def selftest():
    names = ['мать', 'нога', 'бобр']
    dict_words = [('мать', np.zeros((10, 59, 3), np.float32)),
                  ('Нога', np.ones((8, 59, 3), np.float32)),
                  ('кит', np.full((6, 59, 3), 2.0, np.float32))]
    # «кит» не входит в имена → не покрывается; остальные мапятся по классам
    pool, classes, covered = build_dict_pool(dict_words, names)
    assert covered == 2 and classes == [0, 1]
    assert len(pool[0]) == 1 and len(pool[1]) == 1
    # recombine: 2–5 разных классов, кадры и метки согласованы
    pool = {0: [np.zeros((20, 59, 3), np.float32)],
            1: [np.ones((20, 59, 3), np.float32)],
            2: [np.full((20, 59, 3), 2.0, np.float32)]}
    rng = np.random.default_rng(0)
    for _ in range(20):
        s, labels = recombine_dict(pool, [0, 1, 2], rng)
        assert 2 <= len(labels) <= 5 and len(set(labels)) == len(labels)
        assert s.ndim == 3 and s.shape[1] == 59 and len(s) >= 2 * len(labels) + 1
    # mix_dict: на больших пробах не падает и всегда валиден
    phrases = [(np.zeros((40, 59, 3), np.float32), [0, 1])]
    for _ in range(400):
        slovo_pool = {c: [np.full((20, 59, 3), c, np.float32)] for c in range(3)}
        s, lab = mix_dict(pool, [0, 1, 2], slovo_pool,
                          [0, 1, 2] * 3, [], phrases, np.random.default_rng(_),
                          dict_frac=0.3, corpus_frac=0.2, phrase_frac=0.25,
                          single_frac=0.15)
        assert s.ndim == 3 and lab
    # extra-словарь: чистка меток, фильтр мусора, кап на повторы
    p = Path('reports/evidence/_extra_selftest.npz')
    p.parent.mkdir(parents=True, exist_ok=True)
    f = np.zeros((30, 59, 3), np.float32)
    np.savez(p, frames=f, offsets=np.array([0, 10, 20, 30]),
             words=np.array(['- Абрикос', 'Вариант 1', 'абрикос']))
    try:
        ex = load_extra_dict(p)
        # «- Абрикос» и «абрикос» сливаются (2 клипа), «Вариант 1» с цифрой — мусор
        assert [w for w, _ in ex] == ['абрикос', 'абрикос'], [w for w, _ in ex]
    finally:
        p.unlink()
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='D:/hfdata/phrase20/prod/prod.pt')
    ap.add_argument('--dict-data', default='datasets/youtube_dict/rutube_partial.npz')
    ap.add_argument('--extra-dict', default='',
                    help='доп. npz словарей через запятую (lens или offsets)')
    ap.add_argument('--data', default='datasets/slovo/slovo_holistic.npz')
    ap.add_argument('--work', default='D:/hfdata/dict_folds')
    ap.add_argument('--cache', default='D:/hfdata/rutube_phrases/landmarks')
    ap.add_argument('--inventory', default='reports/evidence/rutube_channel_inventory.json')
    ap.add_argument('--out', default='reports/evidence/finetune_dict.json')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--dict-frac', type=float, default=0.30)
    ap.add_argument('--corpus-frac', type=float, default=0.20)
    ap.add_argument('--phrase-frac', type=float, default=0.25)
    ap.add_argument('--single-frac', type=float, default=0.15)
    ap.add_argument('--specaug', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--canon', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--train-all', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)


if __name__ == '__main__':
    main()