"""Дообучение holistic-модели SignNet на YouTube-данных (ARK-словарь + курс).

Стратегия: загружаем чекпойнт slovo_signnet_holistic.pt (1001 класс),
сопоставляем YouTube-слова с классами Slovo через нормализацию, дообучаем
с низким LR на смеси Slovo + YouTube. Слова без совпадения в Slovo пропускаются.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slovo_model import (
    HFEAT,
    SignNet,
    T,
    augment,
    holistic_features,
    resample,
)


def norm(w):
    return w.lower().replace('ё', 'е').strip().strip('.,!?')


OCR_JUNK = {'арк', 'зы'}


def norm_ocr(w):
    """Агрессивная очистка OCR-мусора из ARK-словаря."""
    import re
    w = norm(w)
    w = re.sub(r'\d+арк$', '', w)
    w = re.sub(r'^[а-яе]\s+', '', w)       # «г бабочка» → «бабочка»
    w = re.sub(r'\s+[а-яе]$', '', w)
    w = re.sub(r'[^а-яе\- ]', '', w).strip().strip('-').strip()
    if len(w) < 2 or w in OCR_JUNK:
        return ''
    vowels = sum(1 for c in w if c in 'аеиоуыэюя')
    if vowels == 0 and len(w) > 3:
        return ''
    return w


def class_aliases(names):
    mapping = {}
    for i, name in enumerate(names):
        for word in name.split('/'):
            word = norm(word)
            if word:
                mapping[word] = i if word not in mapping or mapping[word] == i else -1
    return mapping


def load_slovo(path):
    with np.load(path) as d:
        offsets, frames, texts = d['offsets'], d['frames'], d['texts']
        train, users = d['train'], d['users']
    if train.ndim != 1 or len(train) != len(texts) or not np.isin(train, [0, 1]).all():
        raise ValueError(f'Неверная маска train/test: {path}')
    seqs = [frames[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    names = sorted({str(text) for text in texts})
    y = np.array([names.index(text) for text in texts])
    return seqs, y, train.astype(bool), users, names


def file_words(d):
    normalize = norm_ocr if 'variants' in d else norm
    return [normalize(str(w)) for w in d['words']]


def load_samples(path):
    with np.load(path) as d:
        words = file_words(d)
        offsets = d['offsets'] if 'offsets' in d else np.cumsum(np.r_[0, d['lens']])
        frames = d['frames']
    if (offsets.ndim != 1 or len(offsets) != len(words) + 1
            or not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0
            or offsets[-1] != len(frames) or np.any(offsets[1:] < offsets[:-1])
            or np.any(offsets > len(frames))):
        raise ValueError(f'Неверные границы образцов: {path}')
    return [(word, frames[offsets[i]:offsets[i + 1]]) for i, word in enumerate(words)
            if word and offsets[i + 1] - offsets[i] >= 4]


def load_youtube(path, name2idx, add_new=False, allow_new=None, first_new_index=0):
    next_index = max(first_new_index, max(name2idx.values(), default=-1) + 1)
    seqs, labels = [], []
    for word, seq in load_samples(path):
        if word not in name2idx and add_new and (allow_new is None or word in allow_new):
            name2idx[word] = next_index
            next_index += 1
        if word in name2idx and name2idx[word] >= 0:
            seqs.append(seq)
            labels.append(name2idx[word])
    return seqs, np.array(labels, dtype=int)


def file_sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sample_sha256(seq):
    data = np.nan_to_num(np.asarray(seq, dtype='<f4'), copy=True)
    data[data == 0] = 0
    return hashlib.sha256(str(data.shape).encode() + data.tobytes()).hexdigest()


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
    ap.add_argument('--out', required=True, help='новый файл экспериментального checkpoint')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--yt-weight', type=float, default=3.0,
                    help='повтор YouTube-сэмплов (upsampling minority)')
    ap.add_argument('--add-new', action='store_true',
                    help='добавить YouTube-слова, которых нет в Slovo, как новые классы')
    ap.add_argument('--min-samples', type=int, default=3,
                    help='минимум образцов, чтобы завести новый класс (действует с --add-new)')
    a = ap.parse_args()
    if a.epochs < 1 or a.bs < 1 or a.min_samples < 1 or a.lr <= 0 or a.yt_weight < 1:
        ap.error('epochs, bs, min-samples, lr и yt-weight должны быть положительными')
    out_path = Path(a.out)
    eval_path = out_path.with_suffix('.eval.json')
    if out_path.exists() or eval_path.exists():
        ap.error('Выберите новый --out: существующие результаты не перезаписываются')
    torch.manual_seed(a.seed)
    slovo_seqs, slovo_y, slovo_train, _, slovo_names = load_slovo(a.slovo)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('kind') != 'holistic':
        ap.error('Для этих данных нужен holistic checkpoint')
    names = list(ckpt['names'])
    name_indices = {name: i for i, name in enumerate(names)}
    if not set(slovo_names) <= name_indices.keys():
        ap.error('В checkpoint отсутствуют классы Slovo')
    remap = np.array([name_indices[name] for name in slovo_names])
    slovo_y = remap[slovo_y]
    name2idx = class_aliases(names)
    n_slovo_classes = len(names)
    sources = [{'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}
               for path in (a.youtube or [])]

    yt_seqs, yt_y = [], np.array([], dtype=int)
    word_counts = Counter(word for path in (a.youtube or []) for word, _ in load_samples(path))
    total_words = set(word_counts)
    matched_words = {word for word in total_words if name2idx.get(word, -1) >= 0}
    allow_new = {word for word, count in word_counts.items() if count >= a.min_samples} if a.add_new else set()

    for path in (a.youtube or []):
        s, y = load_youtube(path, name2idx, add_new=a.add_new, allow_new=allow_new,
                            first_new_index=n_slovo_classes)
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
        why = f'ниже порога {a.min_samples} обр.' if a.add_new else 'нет в Slovo'
        missing = sorted(not_matched)[:30]
        print(f'  пропущено ({len(not_matched)}, {why}): {", ".join(missing)}...')

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

    if len(tr_seqs) < a.bs:
        ap.error('Размер batch превышает обучающую выборку: обновлений весов не будет')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(tr_seqs) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(steps, 1), pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

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
            print(f'ep {ep:3d} loss {total_loss / max(nb, 1):.3f} '
                  f'[{time.time() - t0:.0f}с]', flush=True)

    inherited = ckpt.get('training', {})
    training = {
        'schema_version': 1,
        'sources': inherited.get('sources', []) + sources,
        'slovo': {'sha256': file_sha256(a.slovo), 'split': 'train'},
        'sample_sha256': sorted(set(inherited.get('sample_sha256', []))
                                | {sample_sha256(seq) for seq in tr_seqs}),
        'base_checkpoint_sha256': file_sha256(a.checkpoint),
        'base_provenance_verified': inherited.get('base_provenance_verified', False),
        'selection': 'fixed_epochs',
        'epochs': a.epochs, 'seed': a.seed, 'lr': a.lr, 'yt_weight': yt_repeat,
        'min_samples': a.min_samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'names': names, 'kind': 'holistic',
                'training': training}, out_path)
    saved = torch.load(out_path, map_location=dev, weights_only=False)
    model.load_state_dict(saved['model'])
    print(f'сохранено {out_path}; оценка сохранённой модели, без выбора эпох по test')
    logits = predict(model, te_seqs, dev)
    yt_logits = predict(model, yt_seqs, dev) if yt_seqs else None
    res = {
        'base_top1': base_acc,
        'final_top1': topk(logits, slovo_y[slovo_te], 1),
        'final_top5': topk(logits, slovo_y[slovo_te], 5),
        'checkpoint_sha256': file_sha256(out_path),
        'checkpoint_selection': 'fixed_epochs',
        'epochs': a.epochs,
        'seed': a.seed,
        'yt_samples': len(yt_seqs),
        'yt_matched_words': len(matched_words),
        'yt_total_words': len(total_words),
        'new_classes': n_new,
        'min_samples': a.min_samples,
        'total_classes': len(names),
    }
    if yt_logits is not None:
        res['yt_train_top1'] = topk(yt_logits, yt_y, 1)
        res['yt_train_top5'] = topk(yt_logits, yt_y, 5)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    eval_path.write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
