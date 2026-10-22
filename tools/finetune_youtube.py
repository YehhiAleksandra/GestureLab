"""Дообучение holistic-модели SignNet на YouTube-данных (ARK-словарь + курс).

Стратегия: загружаем чекпойнт slovo_signnet_holistic.pt (1001 класс),
сопоставляем YouTube-слова с классами Slovo через нормализацию, дообучаем
с низким LR на смеси Slovo + YouTube. Слова без совпадения в Slovo пропускаются.
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
from slovo_model import HFEAT, SignNet, T, augment, holistic_features, resample  # noqa: E402


def norm(w):
    return w.lower().replace('ё', 'е').strip().strip('.,!?')


def norm_ocr(w):
    """Агрессивная очистка OCR-мусора из ARK-словаря."""
    import re
    w = norm(w)
    w = re.sub(r'\d+арк$', '', w)
    w = re.sub(r'^[а-яе]\s+', '', w)       # «г бабочка» → «бабочка»
    w = re.sub(r'\s+[а-яе]$', '', w)
    w = re.sub(r'[^а-яе\- ]', '', w).strip().strip('-').strip()
    if len(w) < 2:
        return ''
    vowels = sum(1 for c in w if c in 'аеиоуыэюя')
    if vowels == 0 and len(w) > 3:
        return ''
    return w


def load_slovo(path):
    d = np.load(path)
    o, frames = d['offsets'], d['frames']
    seqs = [frames[o[i]:o[i + 1]] for i in range(len(o) - 1)]
    names = sorted({str(t) for t in d['texts']})
    y = np.array([names.index(t) for t in d['texts']])
    return seqs, y, d['train'], d['users'], names


def load_youtube(path, name2idx, add_new=False):
    """Загрузка YouTube npz. Если add_new=True, добавляет новые слова в name2idx."""
    d = np.load(path)
    if len(d['words']) == 0:
        return [], np.array([], dtype=int)
    o = d['offsets'] if 'offsets' in d else np.cumsum(np.r_[0, d['lens']])
    frames = d['frames']
    has_variants = 'variants' in d  # ARK data has OCR labels
    seqs, labels = [], []
    for i, w in enumerate(d['words']):
        w = norm_ocr(str(w)) if has_variants else norm(str(w))
        if not w:
            continue
        if w not in name2idx and add_new:
            name2idx[w] = len(name2idx)
        if w in name2idx:
            seg = frames[o[i]:o[i + 1]]
            if len(seg) >= 4:
                seqs.append(seg)
                labels.append(name2idx[w])
    return seqs, np.array(labels, dtype=int)


def batch_feats(seqs, rng=None):
    out = []
    for s in seqs:
        if rng is not None:
            a = rng.uniform(0, 0.15)
            s = resample(augment(s, rng), T, a, 1 - rng.uniform(0, 0.15))
        else:
            s = resample(s, T)
        out.append(holistic_features(s))
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
    ap.add_argument('--slovo', default='datasets/slovo/slovo_holistic.npz')
    ap.add_argument('--youtube', nargs='+', help='npz файлы YouTube (ark_dict.npz, course.npz)')
    ap.add_argument('--checkpoint', default='models/slovo_signnet_holistic.pt')
    ap.add_argument('--out', default='models/slovo_signnet_holistic_yt.pt')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--yt-weight', type=float, default=3.0,
                    help='повтор YouTube-сэмплов (upsampling minority)')
    ap.add_argument('--add-new', action='store_true',
                    help='добавить YouTube-слова, которых нет в Slovo, как новые классы')
    a = ap.parse_args()

    slovo_seqs, slovo_y, slovo_train, slovo_users, names = load_slovo(a.slovo)

    ckpt = torch.load(a.checkpoint, map_location='cpu')
    ckpt_names = ckpt['names']
    if len(ckpt_names) != len(names):
        names = ckpt_names
        name_set = sorted({str(t) for t in np.load(a.slovo, allow_pickle=True)['texts']})
        remap = {name_set.index(t): names.index(t) for t in name_set if t in names}
        slovo_y = np.array([remap.get(int(y), -1) for y in slovo_y])
        keep = slovo_y >= 0
        slovo_seqs = [slovo_seqs[i] for i in range(len(slovo_seqs)) if keep[i]]
        slovo_y = slovo_y[keep]
        slovo_train = slovo_train[keep]
        slovo_users = slovo_users[keep]

    name2idx = {}
    for i, n in enumerate(names):
        for w in n.lower().replace('ё', 'е').strip().split('/'):
            name2idx[w.strip()] = i
    n_slovo_classes = len(ckpt_names)

    yt_seqs, yt_y = [], np.array([], dtype=int)
    total_words = set()
    for path in (a.youtube or []):
        d = np.load(path)
        has_variants = 'variants' in d
        for w in d['words']:
            cleaned = norm_ocr(str(w)) if has_variants else norm(str(w))
            if cleaned:
                total_words.add(cleaned)
    matched_words = {w for w in total_words if w in name2idx}

    for path in (a.youtube or []):
        s, y = load_youtube(path, name2idx, add_new=a.add_new)
        if len(s):
            yt_seqs.extend(s)
            yt_y = np.concatenate([yt_y, y])

    new_classes = {v: k for k, v in name2idx.items() if v >= n_slovo_classes}
    if new_classes:
        names = names + [new_classes[i] for i in range(n_slovo_classes, max(new_classes) + 1)]
    n_new = len(names) - n_slovo_classes

    print(f'Slovo: {sum(slovo_train)} train, {sum(~slovo_train)} test, {n_slovo_classes} классов')
    print(f'YouTube: {len(yt_seqs)} жестов, {len(matched_words)}/{len(total_words)} слов совпали')
    if n_new:
        print(f'  новых классов: {n_new} (всего теперь {len(names)})')
    not_matched = total_words - matched_words - set(new_classes.values())
    if not_matched:
        missing = sorted(not_matched)[:30]
        print(f'  нет в Slovo ({len(not_matched)}): {", ".join(missing)}...')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SignNet(len(names), feat=HFEAT).to(dev)
    sd = ckpt['model']
    if n_new > 0:
        old_w = sd['head.2.weight']
        old_b = sd['head.2.bias']
        sd['head.2.weight'] = torch.cat([old_w, torch.randn(n_new, old_w.shape[1]) * 0.01])
        sd['head.2.bias'] = torch.cat([old_b, torch.zeros(n_new)])
    model.load_state_dict(sd)
    print(f'загружен {a.checkpoint} ({len(names)} классов, kind={ckpt.get("kind")})')

    slovo_tr = np.where(slovo_train)[0]
    slovo_te = np.where(~slovo_train)[0]
    te_seqs = [slovo_seqs[i] for i in slovo_te]

    logits0 = predict(model, te_seqs, dev)
    base_acc = topk(logits0, slovo_y[slovo_te], 1)
    print(f'до дообучения: test top-1 {base_acc:.3f} top-5 {topk(logits0, slovo_y[slovo_te], 5):.3f}')

    tr_seqs = [slovo_seqs[i] for i in slovo_tr]
    tr_y = slovo_y[slovo_tr]
    yt_repeat = max(1, int(a.yt_weight))
    for _ in range(yt_repeat):
        tr_seqs.extend(yt_seqs)
        tr_y = np.concatenate([tr_y, yt_y])
    print(f'обучающая выборка: {len(tr_seqs)} ({sum(slovo_train)} slovo + {len(yt_seqs)}×{yt_repeat} yt)')

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(tr_seqs) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(steps, 1), pct_start=0.1)
    rng = np.random.default_rng(42)
    best, t0 = base_acc, time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        perm = rng.permutation(len(tr_seqs))
        total_loss = 0.0
        nb = 0
        for b in range(len(tr_seqs) // a.bs):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            x = batch_feats([tr_seqs[i] for i in idx], rng).to(dev)
            loss = F.cross_entropy(model(x), torch.as_tensor(tr_y[idx], device=dev), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
            nb += 1
        if ep % 5 == 0 or ep == a.epochs:
            logits = predict(model, te_seqs, dev)
            acc = topk(logits, slovo_y[slovo_te], 1)
            t5 = topk(logits, slovo_y[slovo_te], 5)
            print(f'ep {ep:3d} loss {total_loss / max(nb, 1):.3f} test top1 {acc:.3f} '
                  f'top5 {t5:.3f} [{time.time() - t0:.0f}с]', flush=True)
            if acc > best:
                best = acc
                torch.save({'model': model.state_dict(), 'names': names, 'kind': 'holistic'}, a.out)
                print(f'  сохранено {a.out} (лучший top-1 {best:.3f})')

    if best <= base_acc:
        print(f'дообучение не улучшило модель ({base_acc:.3f} → {best:.3f}), сохраняю последнюю')
        torch.save({'model': model.state_dict(), 'names': names, 'kind': 'holistic'}, a.out)

    logits = predict(model, te_seqs, dev)
    yt_logits = predict(model, yt_seqs, dev) if yt_seqs else None
    res = {
        'base_top1': base_acc,
        'final_top1': topk(logits, slovo_y[slovo_te], 1),
        'final_top5': topk(logits, slovo_y[slovo_te], 5),
        'best_top1': best,
        'yt_samples': len(yt_seqs),
        'yt_matched_words': len(matched_words),
        'yt_total_words': len(total_words),
        'new_classes': n_new,
        'total_classes': len(names),
    }
    if yt_logits is not None:
        res['yt_top1'] = topk(yt_logits, yt_y, 1)
        res['yt_top5'] = topk(yt_logits, yt_y, 5)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    Path('reports/evidence').mkdir(parents=True, exist_ok=True)
    Path('reports/evidence/finetune_youtube_eval.json').write_text(
        json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
