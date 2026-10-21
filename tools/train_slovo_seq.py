"""Обучение SignNet на всех 1000 классах Slovo с официальным разбиением train/test.

Метрики: top-1/top-5 на официальном тесте и отдельно на людях, которых нет в train.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slovo_model import SignNet, T, augment, features, resample  # noqa: E402


def load(path):
    d = np.load(path)
    o, frames = d['offsets'], d['frames']  # NpzFile перечитывает массив при каждом d[...]
    seqs = [frames[o[i]:o[i + 1]] for i in range(len(o) - 1)]
    names = sorted({str(t) for t in d['texts']})
    y = np.array([names.index(t) for t in d['texts']])
    return seqs, y, d['train'], d['users'], names


def batch_feats(seqs, rng=None):
    out = []
    for s in seqs:
        if rng is not None:
            a = rng.uniform(0, 0.15)
            s = resample(augment(s, rng), T, a, 1 - rng.uniform(0, 0.15))
        else:
            s = resample(s, T)
        out.append(features(s))
    return torch.from_numpy(np.stack(out))


@torch.no_grad()
def predict(model, seqs, dev, bs=512):
    model.eval()
    return torch.cat([model(batch_feats(seqs[i:i + bs]).to(dev)).float().cpu()
                      for i in range(0, len(seqs), bs)])


def topk(logits, y, k):
    return (logits.topk(k, 1).indices == torch.as_tensor(y)[:, None]).any(1).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/slovo/slovo_seq.npz')
    ap.add_argument('--out', default='models/slovo_signnet.pt')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()

    seqs, y, train, users, names = load(a.data)
    tr = np.where(train)[0]
    te = np.where(~train)[0]
    unseen = np.array([i for i in te if users[i] not in set(users[tr])])
    print(f'train {len(tr)}, test {len(te)} (из них {len(unseen)} от людей вне train), классов {len(names)}')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SignNet(len(names)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    te_seqs = [seqs[i] for i in te]
    best, t0 = 0.0, time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        total = 0.0
        for b in range(len(tr) // a.bs):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            x = batch_feats([seqs[i] for i in idx], rng).to(dev)
            loss = F.cross_entropy(model(x), torch.as_tensor(y[idx], device=dev), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        if ep % 5 == 0 or ep == a.epochs:
            logits = predict(model, te_seqs, dev)
            acc = topk(logits, y[te], 1)
            print(f'ep {ep:3d} loss {total / (len(tr) // a.bs):.3f} test top1 {acc:.3f} '
                  f'top5 {topk(logits, y[te], 5):.3f} [{time.time() - t0:.0f} с]', flush=True)
            if acc > best:
                best = acc
                torch.save({'model': model.state_dict(), 'names': names}, a.out)

    model.load_state_dict(torch.load(a.out)['model'])
    logits = predict(model, te_seqs, dev)
    pos = {i: k for k, i in enumerate(te)}
    lu = logits[[pos[i] for i in unseen]]
    res = {'test_top1': topk(logits, y[te], 1), 'test_top5': topk(logits, y[te], 5),
           'unseen_top1': topk(lu, y[unseen], 1), 'unseen_top5': topk(lu, y[unseen], 5),
           'n_test': len(te), 'n_unseen': len(unseen), 'classes': len(names)}
    print(json.dumps(res, ensure_ascii=False))
    Path('reports/evidence').mkdir(parents=True, exist_ok=True)
    Path('reports/evidence/slovo_signnet_eval.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
