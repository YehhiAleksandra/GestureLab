"""One-shot центроиды классов по эмбеддингам энкодера CTC — без дообучения.

Зачем: у 120 из 146 слов фразовых клипов один пример, CTC-дообучение учит шаблон
клипа, а не язык (этапы 19–20). Следующий шаг этапа 15, который не делался:
усреднённый эмбеддинг клипа словаря как центроид класса. Камера не нужна —
берём 893 клипа словаря канала (datasets/youtube_dict/rutube_partial.npz) и
кэш ландмарков 67 фразовых клипов. Утечки нет по построению: словарь и фразы —
разные ролики, ни один клип словаря не совпадает с фразовым.

Что делает:
  1. прогоняет каждый клип словаря через энкодер CTC (без головы), маскированное
     среднее по шагам → эмбеддинг [d]; центроид класса = среднее эмбеддингов;
  2. эталонные слова фраз канонизируются к словарю канала (canon_map, этап 20);
  3. каждый фразовый клип → эмбеддинг → top-k ближайших центроидов = мешок слов;
  4. метрики: попадания слов vs CTC greedy той же модели и объединение (union).

  python tools/centroid_eval.py --selftest
  python tools/centroid_eval.py --topk 3
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from eval_phrase_clips import words  # noqa: E402
from finetune_ctc_phrases import canon_map  # noqa: E402
from gloss_coverage import norm, split_classes  # noqa: E402
from sign_stream import load_ctc  # noqa: E402
from sign_utils import split_utterances  # noqa: E402
from train_ctc_slovo import batch_features  # noqa: E402


def load_dict_clips(path):
    """npz словаря → [(слово, кадры)]; слова нормализуются, синонимы не разъединяем."""
    with np.load(path) as d:
        frames, lens = d['frames'], d['lens'].astype(int)
        ws = [norm(str(w)) for w in d['words']]
    offs = np.concatenate([[0], np.cumsum(lens)])
    return [(ws[k], frames[offs[k]:offs[k + 1]].astype(np.float32)) for k in range(len(ws))]


@torch.no_grad()
def embed(model, seqs, feat, dev, bs=64):
    """[(слово, кадры)] → [(слово, эмбеддинг L2-норм.)]; пакеты сортируются по длине."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][1]))
    out = [None] * len(seqs)
    for a in range(0, len(order), bs):
        idx = order[a:a + bs]
        x, lens = batch_features([seqs[i][1] for i in idx], feat)
        h, mask = model.encode(x.to(dev), lens.to(dev))
        w = (~mask).float().unsqueeze(-1) if mask is not None else torch.ones(
            *h.shape[:2], 1, device=h.device)
        e = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = (seqs[i][0], e[j])
    return out


def build_centroids(emb):
    """[(слово, эмбеддинг)] → {слово: центроид}; класс с несколькими клипами усредняется."""
    acc = {}
    for w, e in emb:
        acc.setdefault(w, []).append(e)
    return {w: np.mean(v, 0) for w, v in acc.items()}


def rank(cent_words, cent_mat, e, k):
    """Косинусная близость (вектора нормированы) → top-k слов."""
    sims = cent_mat @ e
    top = np.argsort(-sims)[:k]
    return [cent_words[i] for i in top]


def make_chunks(span, size, stride):
    """Клип → перекрывающиеся чанки по size кадров; хвост покрывается целиком."""
    n = len(span)
    if n <= size:
        return [span]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] + size < n:
        starts.append(n - size)
    return [span[i:i + size] for i in starts]


def chunk_rank(model, span, feat, dev, cent_words, cent_mat, k, size, stride, bs=32):
    """Каждый чанк — возможное одиночное слово: max близости по чанкам → top-k."""
    chunks = make_chunks(span, size, stride)
    embs = [e for _, e in embed(model, [(None, ch) for ch in chunks], feat, dev, bs)]
    sims = np.stack(embs) @ cent_mat.T
    top = np.argsort(-sims.max(0))[:k]
    return [cent_words[i] for i in top]


def match_bag(pred, ref):
    """Мешок слов: сколько предсказанных токенов совпало с эталоном (с кратностью)."""
    return sum((Counter(pred) & Counter(ref)).values())


def load_phrase_spans(cache, inventory):
    """Кэш ландмарков + инвентарь → [(id, фраза, слова, span)]; без фильтра по голове —
    центроиды покрывают весь словарь канала, а не 1000 классов Slovo."""
    inv = json.loads(Path(inventory).read_text(encoding='utf-8'))
    out, skipped = [], []
    for p in inv['phrases']:
        if not p.get('duration'):
            continue
        path = Path(cache) / f"{p['id']}.npz"
        if not path.exists():
            skipped.append({'id': p['id'], 'reason': 'нет кэша'})
            continue
        with np.load(path) as d:
            frames, fps = d['frames'], float(d['fps'])
        spans = split_utterances(frames, fps, pause=0.7, min_len=0.4, max_len=15.0)
        a, b = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(frames))
        ws = words(p['phrase'])
        if not ws:
            skipped.append({'id': p['id'], 'reason': 'пустая фраза'})
            continue
        out.append({'id': p['id'], 'phrase': p['phrase'], 'ref': ws,
                    'span': frames[a:b].astype(np.float32),
                    'used': [int(a), int(b)]})
    return out, skipped


def greedy_words(model, names, span, feat, dev):
    """CTC greedy на том же отрезке → слова классов модели (для сравнения и union)."""
    from sign_stream import greedy_decode
    x, lens = batch_features([span], feat)
    logits = model(x.to(dev), lens.to(dev))
    ids, _ = greedy_decode(logits[0].cpu())
    out = []
    for i in ids:
        out += sorted(split_classes(names[i])) if 0 <= i < len(names) else []
    return [norm(w) for w in out]


def run(a):
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, names, ck = load_ctc(a.model, dev)
    feat = ck.get('feat', 496)

    dict_clips = load_dict_clips(a.dict_data)
    print(f'словарь: {len(dict_clips)} клипов, {len(set(w for w, _ in dict_clips))} слов', flush=True)
    emb = embed(model, dict_clips, feat, dev, a.batch)
    centroids = build_centroids(emb)
    cent_words = sorted(centroids)
    cent_mat = np.stack([centroids[w] for w in cent_words])
    vocab = set(cent_words)
    print(f'центроиды: {len(cent_words)}, устройство {dev}', flush=True)

    clips, skipped = load_phrase_spans(a.cache, a.inventory)
    print(f'клипов: {len(clips)}, пропущено: {len(skipped)}', flush=True)

    t0 = time.time()
    rows = []
    totals = Counter()
    for i, c in enumerate(clips):
        ref_raw = [norm(w) for w in c['ref']]
        cmap = canon_map(ref_raw, vocab)
        ref = [cmap.get(w, w) for w in ref_raw]
        row = {'id': c['id'], 'phrase': c['phrase'], 'ref': ref, 'used': c['used']}
        try:
            (_, e), = embed(model, [(c['id'], c['span'])], feat, dev, 1)
            pred = rank(cent_words, cent_mat, e, a.topk)
            row['pred'] = pred
            row['hits'] = match_bag(pred, ref)
            totals['centroid'] += row['hits']
            if a.chunk:
                cpred = chunk_rank(model, c['span'], feat, dev, cent_words, cent_mat,
                                   a.topk, a.chunk, a.stride)
                row['chunk_pred'] = cpred
                row['chunk_hits'] = match_bag(cpred, ref)
                totals['chunk'] += row['chunk_hits']
            if a.with_ctc:
                gp = greedy_words(model, names, c['span'], feat, dev)
                row['ctc'] = gp
                row['ctc_hits'] = match_bag(gp, ref)
                row['union_hits'] = match_bag(list(dict.fromkeys(gp + pred)), ref)
                totals['ctc'] += row['ctc_hits']
                totals['union'] += row['union_hits']
        except Exception as ex:  # без ретраев: сбой остаётся в отчёте
            row['error'] = f'{type(ex).__name__}: {ex}'
        totals['ref'] += len(ref)
        rows.append(row)
        print(f"{i + 1}/{len(clips)} {c['id'][:8]} {c['phrase']!r}: "
              f"{row.get('pred', row.get('error'))}", flush=True)

    summary = {
        'clips': len(rows), 'failed': sum(1 for r in rows if r.get('error')),
        'ref_tokens': totals['ref'],
        'centroid_hits': totals['centroid'],
        'centroid_recall': round(totals['centroid'] / max(totals['ref'], 1), 3),
        'chunk_hits': totals['chunk'],
        'chunk_recall': round(totals['chunk'] / max(totals['ref'], 1), 3),
        'topk': a.topk, 'centroids': len(cent_words),
        'skipped': len(skipped), 'seconds': round(time.time() - t0, 1),
    }
    if a.with_ctc:
        summary['ctc_greedy_hits'] = totals['ctc']
        summary['union_hits'] = totals['union']
        summary['union_recall'] = round(totals['union'] / max(totals['ref'], 1), 3)
    result = {'model': Path(a.model).name, 'dict_data': str(a.dict_data),
              'date': time.strftime('%Y-%m-%d'), 'summary': summary,
              'skipped': skipped, 'clips': rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f'→ {a.out}', flush=True)


def selftest():
    # центроиды: два клипа одного слова усредняются, разные слова не смешиваются
    e1, e2, e3 = np.eye(3, dtype=np.float32)
    cents = build_centroids([('рука', e1), ('рука', e2), ('нога', e3)])
    assert set(cents) == {'рука', 'нога'} and np.allclose(cents['рука'], (e1 + e2) / 2)
    # ранжирование: ближайший — «рука», затем «нога»
    cw = sorted(cents)
    cm = np.stack([cents[w] for w in cw])
    assert rank(cw, cm, e1 + 0.01 * e2, 2) == ['рука', 'нога']
    # мешок с кратностью: два «вас» в эталоне, одно в предсказании
    assert match_bag(['вас', 'я'], ['вас', 'вас', 'я']) == 2
    assert match_bag(['вас', 'вас'], ['вас']) == 1
    # канонизация подмешивается: «автомобилем» → «автомобиль», если он в словаре
    cmap = canon_map(['автомобилем'], {'автомобиль'})
    assert cmap['автомобилем'] == 'автомобиль'
    # нормализация слова словаря: капитализация и пунктуация срезаются
    assert norm('Автобус ') == 'автобус'
    # чанки: короткий клип не режется, длинный покрывается с перекрытием
    long = np.zeros((40, 59, 3), np.float32)
    parts = make_chunks(long, 32, 16)
    assert len(parts) == 2 and len(parts[1]) == 32
    assert np.shares_memory(parts[1][0], long[8])
    assert len(make_chunks(long[:10], 32, 16)) == 1
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='D:/hfdata/phrase20/prod/prod.pt')
    ap.add_argument('--dict-data', default='datasets/youtube_dict/rutube_partial.npz')
    ap.add_argument('--inventory', default='reports/evidence/rutube_channel_inventory.json')
    ap.add_argument('--cache', default='D:/hfdata/rutube_phrases/landmarks')
    ap.add_argument('--out', default='reports/evidence/centroid_eval.json')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--chunk', type=int, default=0,
                    help='кадров в чанке (0 — только усреднение по всему клипу)')
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='')
    ap.add_argument('--with-ctc', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)


if __name__ == '__main__':
    main()