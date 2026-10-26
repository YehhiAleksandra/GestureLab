"""Обучение CTC-распознавателя непрерывного РЖЯ на потоках из склеенных клипов Slovo.

Потоки собираются двумя способами: случайные 2–6 жестов (покрытие всех 1000 классов) и
реальные цепочки глоссов из корпуса parallel_corpus (порядок и сочетаемость как в языке).
Клипы проходят поворот/зеркало/шум, случайную скорость, линейный переход, иногда паузу с
опущенными руками; признаки — SpecAugment. Дополнительно вспомогательная CE на изолированных
клипах, чтобы одиночный жест не терялся за CTC.

Протокол: заранее фиксированное число эпох, сохранение итогового (не лучшего на тесте)
checkpoint, повторная загрузка и оценка на отдельном наборе потоков из тестовых клипов
(случайные и корпусные). Манифест хранит хеш датасета и параметры прогона.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gloss_to_text import load_corpus
from train_slovo_seq import load

from sign_stream import CTCSignNet, greedy_decode, load_ctc, out_lengths
from sign_utils import FEAT, FEAT_MOTION, features
from slovo_model import augment, resample


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _transition(prev, s, rng, pause_prob, blend_prob):
    """Стык двух жестов: пауза с опущенными руками или линейное перетекание формы."""
    if prev is None:
        return []
    if rng.random() < pause_prob:
        k = int(rng.integers(4, 15))
        neutral = np.full((k, s.shape[1], 3), np.nan, np.float32)
        neutral[:, 42:] = prev[-1, 42:]  # поза и лицо стоят, руки опущены
        return [neutral]
    if rng.random() < blend_prob:
        k = int(rng.integers(2, 6))
        mix = np.linspace(0, 1, k + 2)[1:-1, None, None].astype(np.float32)
        return [prev[-1] * (1 - mix) + s[0] * mix]
    return []


def make_stream(seqs_by_class, classes, rng, augment_clips=True, pause_prob=0.25,
                blend_prob=0.6, n_range=(2, 6)):
    """Склеить случайные клипы в непрерывный поток. Возвращает (кадры, классы)."""
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    chosen = rng.choice(classes, n, replace=False)
    parts, labels, prev = [], [], None
    for c in chosen:
        pool = seqs_by_class[int(c)]
        s = pool[int(rng.integers(len(pool)))]
        if augment_clips:
            s = augment(s, rng)
        s = resample(s, int(rng.integers(16, 49)))
        parts += _transition(prev, s, rng, pause_prob, blend_prob)
        parts.append(s)
        labels.append(int(c))
        prev = s
    return np.concatenate(parts), labels


def corpus_phrases(index, split='train'):
    """Цепочки глоссов корпуса → индексы классов; фразы с незнакомыми словами отбрасываются."""
    out = []
    for p in load_corpus(split):
        ids = [index.get(w.lower()) for w in p['rsl'].split()]
        if ids and all(i is not None for i in ids):
            out.append(ids)
    return out


def make_corpus_stream(seqs_by_class, phrases, rng, augment_clips=True, pause_prob=0.2,
                       blend_prob=0.5, dur=(14, 56)):
    """Поток по реальной цепочке глоссов: порядок и сочетаемость из корпуса."""
    phrase = phrases[int(rng.integers(len(phrases)))]
    parts, prev = [], None
    for c in phrase:
        pool = seqs_by_class[c]
        s = pool[int(rng.integers(len(pool)))]
        if augment_clips:
            s = augment(s, rng)
        s = resample(s, int(rng.integers(dur[0], dur[1] + 1)))
        parts += _transition(prev, s, rng, pause_prob, blend_prob)
        parts.append(s)
        prev = s
    return np.concatenate(parts), list(phrase)


def spec_augment(x, lens, rng, n_time=2, n_feat=2, frac=0.08):
    """Маски по времени и признакам; пропуск заполняется средним по клипу."""
    x = x.clone()
    for i in range(x.shape[0]):
        t = int(lens[i])
        if t < 2:
            continue
        mean = x[i, :t].mean(0)
        for _ in range(n_time):
            w = max(1, int(rng.integers(1, max(2, int(frac * t)))))
            s = int(rng.integers(0, t - w + 1))
            x[i, s:s + w] = mean
        for _ in range(n_feat):
            w = max(1, int(rng.integers(1, max(2, int(frac * x.shape[2])))))
            s = int(rng.integers(0, x.shape[2] - w + 1))
            x[i, :t, s:s + w] = mean[s:s + w]
    return x


def aux_batch(seqs_by_class, classes, rng, n, feat=FEAT):
    """Случайные изолированные клипы для вспомогательной классификации."""
    streams, labels = [], []
    for _ in range(n):
        c = int(rng.choice(classes))
        pool = seqs_by_class[c]
        s = augment(pool[int(rng.integers(len(pool)))], rng)
        streams.append(resample(s, int(rng.integers(16, 49))))
        labels.append(c)
    x, lens = batch_features(streams, feat)
    return x, lens, torch.tensor(labels, dtype=torch.long)


def batch_features(streams, feat=FEAT):
    feats = [features(s, feat) for s in streams]
    lens = np.array([len(f) for f in feats])
    x = np.zeros((len(feats), lens.max(), feat), np.float32)
    for i, f in enumerate(feats):
        x[i, :len(f)] = f
    return torch.from_numpy(x), torch.from_numpy(lens)


def edit_counts(pred, ref):
    """Левенштейн: (замены, вставки, удаления)."""
    n, m = len(ref), len(pred)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != pred[j - 1]))
    return int(d[n, m])


def prepare_slovo(path):
    """Клипы → классы без no_event, train/test как в официальном разбиении Slovo."""
    seqs, y, train, _users, all_names = load(path)
    keep = [i for i, n in enumerate(all_names) if n != 'no_event']
    remap = {old: new for new, old in enumerate(keep)}
    names = [all_names[i] for i in keep]
    seqs_by_class, te_by_class = {}, {}
    for i in np.where(train)[0]:
        if int(y[i]) in remap:
            seqs_by_class.setdefault(remap[int(y[i])], []).append(seqs[i])
    for i in np.where(~train)[0]:
        if int(y[i]) in remap:
            te_by_class.setdefault(remap[int(y[i])], []).append(seqs[i])
    return {'names': names, 'seqs_by_class': seqs_by_class, 'te_by_class': te_by_class,
            'classes': sorted(seqs_by_class), 'te_classes': sorted(te_by_class),
            'n_train': sum(len(v) for v in seqs_by_class.values()),
            'n_test': sum(len(v) for v in te_by_class.values())}


def samples_random(seqs_by_class, classes, rng, n=100, pauses=False):
    return [make_stream(seqs_by_class, classes, rng, augment_clips=False,
                        pause_prob=1.0 if pauses else 0.0, blend_prob=1.0) for _ in range(n)]


def samples_corpus(by_class, phrases, rng, n=200, pauses=False):
    return [make_corpus_stream(by_class, phrases, rng, augment_clips=False,
                               pause_prob=1.0 if pauses else 0.0, blend_prob=1.0)
            for _ in range(n)]


def sample_item(seqs_by_class, classes, corpus_train, rng, corpus_frac=0.6, single_frac=0.0):
    """Один обучающий пример: цепочка корпуса, одиночный клип или случайный поток."""
    r = rng.random()
    if corpus_train and r < corpus_frac:
        return make_corpus_stream(seqs_by_class, corpus_train, rng)
    if single_frac > 0 and r < corpus_frac + single_frac:
        c = int(rng.choice(classes))
        s = augment(seqs_by_class[c][int(rng.integers(len(seqs_by_class[c])))], rng)
        return resample(s, int(rng.integers(16, 49))), [c]
    return make_stream(seqs_by_class, classes, rng)


@torch.no_grad()
def metrics_of(model, samples, dev='cpu'):
    """Greedy-метрики на готовом наборе «поток + эталонные классы»."""
    model.eval()
    pos_ok = total = exact = dist = 0
    for stream, labels in samples:
        x, lengths = batch_features([stream], model.inp[0].in_features)
        pred, _ = greedy_decode(model(x.to(dev), lengths.to(dev))[0].cpu())
        pos_ok += sum(a == b for a, b in zip(pred, labels))
        total += len(labels)
        exact += int(pred == labels)
        dist += edit_counts(pred, labels)
    return {'gloss_acc': pos_ok / total, 'exact_acc': exact / len(samples),
            'edit_rate': dist / total, 'n_streams': len(samples), 'n_glosses': total}


def evaluate(model, seqs_by_class, classes, rng, n=100, pauses=False, dev='cpu'):
    return metrics_of(model, samples_random(seqs_by_class, classes, rng, n, pauses), dev)


def evaluate_corpus(model, by_class, phrases, rng, n=200, pauses=False, dev='cpu'):
    return metrics_of(model, samples_corpus(by_class, phrases, rng, n, pauses), dev)


def warm_start(model, path, names):
    """Взять форму кистей и энкодер из готовой модели; blank ← no_event.

    Годится и для продолжения обучения с CTC-чекпойнта: переносятся все общие слои,
    включая позиционное кодирование, если формы совпадают.
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    src, src_names = ck['model'], ck['names']
    dst = model.state_dict()
    for k in dst:
        if (k in src and k.startswith(('inp.', 'conv.', 'down.', 'enc.', 'pos.'))
                and src[k].shape == dst[k].shape):
            dst[k] = src[k]
    rows = {n: i for i, n in enumerate(src_names)}
    head = torch.zeros_like(dst['head.weight'])
    bias = torch.zeros_like(dst['head.bias'])
    hw, hb = src.get('head.2.weight', src.get('head.weight')), src.get('head.2.bias',
                                                                     src.get('head.bias'))
    ctc_src = hw.shape[0] == len(src_names) + 1  # CTC: строка 0 — blank, класс c — строка c+1
    for j, n in enumerate(names, 1):
        if n in rows:
            r = rows[n] + 1 if ctc_src else rows[n]
            head[j], bias[j] = hw[r], hb[r]
    if ctc_src:
        head[0], bias[0] = hw[0], hb[0]
    elif 'no_event' in rows:
        head[0], bias[0] = hw[rows['no_event']], hb[rows['no_event']]
    dst['head.weight'], dst['head.bias'] = head, bias
    model.load_state_dict(dst)
    print(f'warm start {path}: классов перенесено {sum(n in rows for n in names)}/{len(names)}',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/slovo/slovo_holistic.npz')
    ap.add_argument('--out', default='models/slovo_ctc_stream.pt')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--steps', type=int, default=200, help='шагов на эпоху')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--init', help='чекпойнт для тёплого старта или продолжения обучения')
    ap.add_argument('--corpus-frac', type=float, default=0.6,
                    help='доля потоков из цепочек корпуса (остальные — случайные классы)')
    ap.add_argument('--single-frac', type=float, default=0.0,
                    help='доля CTC-примеров из одиночных клипов (0 — как раньше)')
    ap.add_argument('--d', type=int, default=256, help='ширина энкодера')
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--aux-weight', type=float, default=0.3,
                    help='вес вспомогательной CE на изолированных клипах; 0 — выключить')
    ap.add_argument('--aux-bs', type=int, default=16)
    ap.add_argument('--specaug', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--motion', action='store_true',
                    help='добавить к признакам кадра его изменение (FEAT_MOTION вместо FEAT)')
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f'{a.out} уже существует — выберите другое имя')

    d = prepare_slovo(a.data)
    names, seqs_by_class = d['names'], d['seqs_by_class']
    te_by_class, classes, te_classes = d['te_by_class'], d['classes'], d['te_classes']
    print(f'классов {len(names)}, train клипов {d["n_train"]}, test клипов {d["n_test"]}, '
          f'классов в тесте {len(te_classes)}', flush=True)

    corpus_train, corpus_test = [], []
    if a.corpus_frac > 0:
        index = {n.lower(): i for i, n in enumerate(names)}
        try:
            corpus_train = corpus_phrases(index, 'train')
            corpus_test = [p for p in corpus_phrases(index, 'test')
                           if all(c in te_by_class for c in p)]
        except Exception as e:  # noqa: BLE001 — внешний сервис: любая ошибка → случайные потоки
            print(f'корпус недоступен ({e}) — только случайные потоки', flush=True)
    print(f'корпус: train {len(corpus_train)} фраз, test {len(corpus_test)} фраз', flush=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    aux_n = len(names) if a.aux_weight > 0 else 0
    feat_in = FEAT_MOTION if a.motion else FEAT
    model = CTCSignNet(len(names), d=a.d, layers=a.layers, dropout=a.dropout, feat=feat_in,
                       aux=aux_n).to(dev)
    if a.init:
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
                s, lab = sample_item(seqs_by_class, classes, corpus_train, rng,
                                     corpus_frac=a.corpus_frac, single_frac=a.single_frac)
                streams.append(s)
                labels.append(lab)
            x, lens = batch_features(streams, feat_in)
            if a.specaug:
                x = spec_augment(x, lens.numpy(), rng)
            # PyTorch CTCLoss берёт цели как индексы логитов напрямую (0 = blank), без сдвига
            targets = torch.tensor([c + 1 for lab in labels for c in lab], dtype=torch.long)
            tlens = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
            logits = model(x.to(dev), lens.to(dev))
            logp = logits.log_softmax(-1).transpose(0, 1)
            loss = ctc(logp, targets.to(dev), torch.from_numpy(out_lengths(lens.numpy())).to(dev),
                       tlens.to(dev))
            if aux_n:
                ax, alens, ay = aux_batch(seqs_by_class, classes, rng, a.aux_bs, feat_in)
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
            line = (f'ep {ep:3d} loss {total / a.steps:.3f} | clean gloss {ev["gloss_acc"]:.3f} '
                    f'exact {ev["exact_acc"]:.3f} edit {ev["edit_rate"]:.3f}')
            if corpus_test:
                ec = evaluate_corpus(model, te_by_class, corpus_test, np.random.default_rng(1234),
                                     n=60, dev=dev)
                line += f' | corpus gloss {ec["gloss_acc"]:.3f} edit {ec["edit_rate"]:.3f}'
            print(line + f' [{time.time() - t0:.0f} с]', flush=True)

    arch = {'d': a.d, 'layers': a.layers, 'heads': 4, 'dropout': a.dropout}
    config = {'data': a.data, 'epochs': a.epochs, 'steps': a.steps, 'bs': a.bs, 'lr': a.lr,
              'seed': a.seed, 'n_train': d['n_train'], 'n_classes': len(names),
              'corpus_frac': a.corpus_frac, 'single_frac': a.single_frac,
              'aux_weight': a.aux_weight, 'aux_bs': a.aux_bs,
              'specaug': a.specaug, 'motion': a.motion, 'n_corpus_train': len(corpus_train),
              'n_corpus_test': len(corpus_test), 'init': a.init}
    torch.save({'model': model.state_dict(), 'names': names, 'feat': feat_in, 'arch': arch,
                'aux': aux_n, 'config': config, 'kind': 'ctc_holistic'}, a.out)

    model2, names2, _ = load_ctc(a.out, dev)
    assert names2 == names
    clean = metrics_of(model2, samples_random(te_by_class, te_classes,
                                              np.random.default_rng(1234), 200), dev)
    paused = metrics_of(model2, samples_random(te_by_class, te_classes,
                                               np.random.default_rng(1234), 200, pauses=True), dev)
    res = {'clean': clean, 'with_pauses': paused, 'checkpoint': Path(a.out).name,
           'checkpoint_sha256': file_sha256(a.out)}
    if corpus_test:
        res['corpus'] = metrics_of(model2, samples_corpus(
            te_by_class, corpus_test, np.random.default_rng(1234), 200), dev)
        res['corpus_with_pauses'] = metrics_of(model2, samples_corpus(
            te_by_class, corpus_test, np.random.default_rng(1234), 200, pauses=True), dev)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    Path('reports/evidence').mkdir(parents=True, exist_ok=True)
    Path(f'reports/evidence/{Path(a.out).stem}_eval.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    manifest = {'schema_version': 1, 'kind': 'ctc_holistic',
                'dataset': a.data, 'dataset_sha256': file_sha256(a.data),
                'checkpoint_sha256': res['checkpoint_sha256'],
                'init_sha256': file_sha256(a.init) if a.init else None,
                'names_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                'config': config, 'arch': arch, 'git_commit': git_commit(),
                'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    Path(str(a.out) + '.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'сохранено {a.out} и манифест', flush=True)


if __name__ == '__main__':
    main()
