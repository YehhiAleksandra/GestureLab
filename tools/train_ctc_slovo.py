"""Обучение CTC-распознавателя непрерывного РЖЯ на потоках из склеенных клипов Slovo.

Изолированные клипы Slovo склеиваются в последовательности 2–6 жестов: со сдвигом скорости,
поворотом/зеркалом/шумом, линейным переходом между жестами и иногда паузой с опущенными
руками. Модель учится выдавать глоссы в порядке показа и сама расставляет границы.

Протокол: заранее фиксированное число эпох, сохранение итогового (не лучшего на тесте)
checkpoint, повторная загрузка и оценка на отдельном наборе синтетических потоков из
тестовых клипов. Манифест хранит хеш датасета и параметры прогона.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_slovo_seq import load

from sign_stream import CTCSignNet, greedy_decode, load_ctc, out_lengths
from sign_utils import FEAT, stream_features
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
    except Exception:
        return None


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
        if prev is not None:
            if rng.random() < pause_prob:
                k = int(rng.integers(4, 15))
                neutral = np.full((k, s.shape[1], 3), np.nan, np.float32)
                neutral[:, 42:] = prev[-1, 42:]  # поза и лицо стоят, руки опущены
                parts.append(neutral)
            elif rng.random() < blend_prob:
                k = int(rng.integers(2, 6))
                mix = np.linspace(0, 1, k + 2)[1:-1, None, None].astype(np.float32)
                parts.append(prev[-1] * (1 - mix) + s[0] * mix)
        parts.append(s)
        labels.append(int(c))
        prev = s
    return np.concatenate(parts), labels


def batch_features(streams):
    feats = [stream_features(s) for s in streams]
    lens = np.array([len(f) for f in feats])
    x = np.zeros((len(feats), lens.max(), FEAT), np.float32)
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


@torch.no_grad()
def evaluate(model, seqs_by_class, classes, rng, n=100, pauses=False, dev='cpu'):
    model.eval()
    pos_ok = total = exact = dist = 0
    for _ in range(n):
        stream, labels = make_stream(seqs_by_class, classes, rng, augment_clips=False,
                                     pause_prob=1.0 if pauses else 0.0, blend_prob=1.0)
        x, lengths = batch_features([stream])
        pred, _ = greedy_decode(model(x.to(dev), lengths.to(dev))[0].cpu())
        pos_ok += sum(a == b for a, b in zip(pred, labels))
        total += len(labels)
        exact += int(pred == labels)
        dist += edit_counts(pred, labels)
    return {'gloss_acc': pos_ok / total, 'exact_acc': exact / n,
            'edit_rate': dist / total, 'n_streams': n, 'n_glosses': total}


def warm_start(model, path, names):
    """Взять форму кистей и энкодер из изолированной модели; blank ← no_event."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    src, src_names = ck['model'], ck['names']
    dst = model.state_dict()
    for k in dst:
        if k in src and k.startswith(('inp.', 'conv.', 'enc.')):
            dst[k] = src[k]
    rows = {n: i for i, n in enumerate(src_names)}
    head = torch.zeros_like(dst['head.weight'])
    bias = torch.zeros_like(dst['head.bias'])
    for j, n in enumerate(names, 1):
        if n in rows:
            head[j], bias[j] = src['head.2.weight'][rows[n]], src['head.2.bias'][rows[n]]
    if 'no_event' in rows:
        head[0], bias[0] = src['head.2.weight'][rows['no_event']], src['head.2.bias'][rows['no_event']]
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
    ap.add_argument('--init', help='чекпойнт изолированной модели для тёплого старта')
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f'{a.out} уже существует — выберите другое имя')

    seqs, y, train, _users, all_names = load(a.data)
    keep = [i for i, n in enumerate(all_names) if n != 'no_event']
    remap = {old: new for new, old in enumerate(keep)}
    names = [all_names[i] for i in keep]
    tr = [i for i in np.where(train)[0] if int(y[i]) in remap]
    te = [i for i in np.where(~train)[0] if int(y[i]) in remap]
    seqs_by_class = {}
    for i in tr:
        seqs_by_class.setdefault(remap[int(y[i])], []).append(seqs[i])
    te_by_class = {}
    for i in te:
        te_by_class.setdefault(remap[int(y[i])], []).append(seqs[i])
    classes = sorted(seqs_by_class)
    te_classes = sorted(c for c in te_by_class if len(te_by_class[c]) >= 1)
    print(f'классов {len(names)}, train клипов {len(tr)}, test клипов {len(te)}, '
          f'классов в тесте {len(te_classes)}', flush=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = CTCSignNet(len(names)).to(dev)
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
                s, lab = make_stream(seqs_by_class, classes, rng)
                streams.append(s)
                labels.append(lab)
            x, lens = batch_features(streams)
            # PyTorch CTCLoss берёт цели как индексы логитов напрямую (0 = blank), без сдвига
            targets = torch.tensor([c + 1 for lab in labels for c in lab], dtype=torch.long)
            tlens = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
            logits = model(x.to(dev), lens.to(dev))
            logp = logits.log_softmax(-1).transpose(0, 1)
            loss = ctc(logp, targets.to(dev), torch.from_numpy(out_lengths(lens.numpy())).to(dev),
                       tlens.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss)
        if ep % a.eval_every == 0 or ep == a.epochs:
            ev = evaluate(model, te_by_class, te_classes, np.random.default_rng(1234),
                          n=60, dev=dev)
            print(f'ep {ep:3d} loss {total / a.steps:.3f} | clean gloss {ev["gloss_acc"]:.3f} '
                  f'exact {ev["exact_acc"]:.3f} edit {ev["edit_rate"]:.3f} '
                  f'[{time.time() - t0:.0f} с]', flush=True)

    arch = {'d': 256, 'layers': 4, 'heads': 4, 'dropout': 0.15}
    config = {'data': a.data, 'epochs': a.epochs, 'steps': a.steps, 'bs': a.bs, 'lr': a.lr,
              'seed': a.seed, 'n_train': len(tr), 'n_classes': len(names)}
    torch.save({'model': model.state_dict(), 'names': names, 'feat': FEAT, 'arch': arch,
                'config': config, 'kind': 'ctc_holistic'}, a.out)

    model2, names2, _ = load_ctc(a.out, dev)
    assert names2 == names
    clean = evaluate(model2, te_by_class, te_classes, np.random.default_rng(1234), n=200, dev=dev)
    paused = evaluate(model2, te_by_class, te_classes, np.random.default_rng(1234), n=200,
                      pauses=True, dev=dev)
    res = {'clean': clean, 'with_pauses': paused, 'checkpoint': Path(a.out).name,
           'checkpoint_sha256': file_sha256(a.out)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    Path('reports/evidence').mkdir(parents=True, exist_ok=True)
    Path(f'reports/evidence/{Path(a.out).stem}_eval.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    manifest = {'schema_version': 1, 'kind': 'ctc_holistic',
                'dataset': a.data, 'dataset_sha256': file_sha256(a.data),
                'checkpoint_sha256': res['checkpoint_sha256'], 'names_sha256': hashlib.sha256(
                    '\n'.join(names).encode()).hexdigest(),
                'config': config, 'arch': arch, 'git_commit': git_commit(),
                'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    Path(str(a.out) + '.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'сохранено {a.out} и манифест', flush=True)


if __name__ == '__main__':
    main()
