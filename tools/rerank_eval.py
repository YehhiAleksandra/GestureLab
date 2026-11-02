"""Реранк n-best цепочек CTC центроидной близостью — фьюжн двух каналов (этап 21, п.1).

Зачем: этап 20 показал потолок 45/196 у production-цепочки на фразах; этап 21 дал
независимый сигнал (чанк-центроиды по словарю канала, 20.4% recall без обучения).
Здесь оба канала объединяются: n-best цепочки beam-декодера фолд-моделей
переупорядочиваются суммой «средняя близость слов цепочки к центроидам − λ·ранг»,
плюс мешковые варианты (объединение слов цепочек, голосование, добавление
центроидных кандидатов). Декод запускается один раз на клип, перебор λ — офлайн.

Честность: фолды `folds(n, 5, 2026)`, канонизация по именам фолд-головы, LM выключена
(лучший режим этапа 20) — но независимость держится только на --work из фолдов,
которые не видели клипов оценки, и на --prod-энкодере без фразового обучения. Дефолт
--prod = phrase20/prod/prod.pt (обучен на всех 67 клипах), поэтому каналы rerank,
bag_cent и aff_top считаются с утечкой (B62); чистые — baseline, bag, votes.

  python tools/rerank_eval.py --selftest
  python tools/rerank_eval.py --nbest 20
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

from centroid_eval import (build_centroids, chunk_rank, embed,  # noqa: E402
                           load_dict_clips, make_chunks)
from dialogue_translator import Recognizer  # noqa: E402
from finetune_ctc_phrases import canon_map, folds, head_names, load_phrase_clips  # noqa: E402
from gloss_coverage import split_classes  # noqa: E402
from sign_stream import load_ctc  # noqa: E402
from slovo_model import resample  # noqa: E402
from eval_phrase_clips import words  # noqa: E402


def chain_bag(chains, top=10):
    """Первые top цепочек → мешок слов: кратность слова = максимум по цепочкам."""
    bag = Counter()
    for ch in chains[:top]:
        c = Counter(w.lower() for w in ch)
        for w, n in c.items():
            bag[w] = max(bag[w], n)
    return bag


def votes_bag(chains, top=10, need=2):
    """Слова, встретившиеся минимум в need цепочках из первых top."""
    cnt = Counter()
    for ch in chains[:top]:
        for w in {w.lower() for w in ch}:
            cnt[w] += 1
    return Counter({w: n for w, n in cnt.items() if n >= need})


def bag_hits(bag, ref):
    """Пересечение мешка предсказания с мешком эталона (с кратностью)."""
    return sum((bag & Counter(ref)).values())


def word_affinity(chain, aff, default):
    """Средняя близость слов цепочки к центроидам; синонимы «а/б» → max по частям."""
    vals = []
    for w in {x.lower() for x in chain}:
        parts = [p for p in split_classes(w) if p in aff]
        if parts:
            vals.append(max(aff[p] for p in parts))
    return float(np.mean(vals)) if vals else default


def rerank(chains, affs, default, lam):
    """Индекс лучшей цепочки: средняя близость − λ·ранг (0 = CTC top-1)."""
    return int(np.argmax([word_affinity(ch, affs, default) - lam * i
                          for i, ch in enumerate(chains)]))


def merge_bags(*bags):
    """Объединение мешков: кратность = максимум по источникам."""
    out = Counter()
    for b in bags:
        for w, n in b.items():
            out[w] = max(out[w], n)
    return out


def tta_chains(rec, span, speeds):
    """Декод на нескольких темпах: цепочки всех вариантов в один пул (TTA).

    Темп s<1 растягивает клип, s>1 сжимает; обучение этапа 20 уже видело темп
    0.8–1.25, поэтому варианты остаются в домене модели. Первый вариант —
    исходный клип (его top-1 остаётся baseline).
    """
    variants, base = [], None
    for i, s in enumerate(speeds):
        span_s = span if s == 1.0 else resample(
            span, t=max(8, round(len(span) / s)))
        _, ch = rec.decode(span_s)
        if i == 0:
            base = ch or []
        variants.append(ch or [])
    return base, variants


def chunk_affinities(model, feat, dev, cent_words, cent_mat, span, size, stride):
    """Чанки клипа × центроиды → {слово: max близость по чанкам}."""
    chunks = make_chunks(span, size, stride)
    embs = [e for _, e in embed(model, [(None, ch) for ch in chunks], feat, dev, 32)]
    sims = np.stack(embs) @ cent_mat.T
    agg = sims.max(0)
    return {w: float(agg[i]) for i, w in enumerate(cent_words)}


def run(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    work = Path(a.work)
    ck = torch.load(work / 'fold0.pt', map_location='cpu', weights_only=False)
    names = list(ck['names'])

    inv = json.loads(Path(a.inventory).read_text(encoding='utf-8'))
    plain = [w for p in inv['phrases'] if p.get('duration') for w in words(p['phrase'])]
    canon = canon_map(plain, head_names(names)) if a.canon else {}
    clips, skipped = load_phrase_clips(a.cache, a.inventory, names, canon)
    print(f'канонизация: {sum(1 for w in plain if canon.get(w, w) != w)} слов; '
          f'клипов {len(clips)}, пропущено {len(skipped)}', flush=True)

    # центроиды словаря канала: энкодер prod.pt, чанки как в этапе 21
    pmodel, _, cck = load_ctc(a.prod, dev)
    feat = cck.get('feat', 496)
    dict_clips = load_dict_clips(a.dict_data)
    cents = build_centroids(embed(pmodel, dict_clips, feat, dev, a.batch))
    cent_words = sorted(cents)
    cent_mat = np.stack([cents[w] for w in cent_words])
    print(f'центроидов: {len(cent_words)}', flush=True)

    parts = folds(len(clips), a.folds, a.seed)
    speeds = [float(x) for x in a.tta.split(',')]
    lam_grid = [float(x) for x in a.lam.split(',')]
    variants = ['baseline', 'rerank', 'bag', 'votes', 'bag_cent']
    totals = {v: 0 for v in variants}
    pred_n = {v: 0 for v in variants}
    lam_totals = {lm: 0 for lm in lam_grid}
    rows = []

    t0 = time.time()
    for k, held_idx in enumerate(parts):
        held = [clips[i] for i in held_idx]
        rec = Recognizer(str(work / f'fold{k}.pt'), decoder='beam', beam=a.beam,
                         lm_weight=0.0, word_bonus=a.word_bonus, smooth=1, nbest=a.nbest)
        for c in held:
            base_chains, tta_lists = tta_chains(rec, c['span'], speeds)
            tta_variants = [v for v in tta_lists if v] or [base_chains]
            if not base_chains:
                base_chains = [[g for g, _ in rec.decode(c['span'])[0]]]
            chains = base_chains  # базовые цепочки: реранк и кривая λ — без TTA
            ref = [canon.get(w, w) for w in c['words']]
            aff = chunk_affinities(pmodel, feat, dev, cent_words, cent_mat,
                                   c['span'], a.chunk, a.stride)
            default = float(np.mean(list(aff.values())))
            bags = {
                'baseline': chain_bag(base_chains[:1]),
                'rerank': chain_bag([chains[rerank(chains, aff, default, a.lam_rank)]]),
                # TTA: мешок per-вариант и объединение — иначе варианты из пула
                # не попадают в первые top
                'bag': merge_bags(*[chain_bag(v, a.top) for v in tta_variants]),
                'votes': merge_bags(*[votes_bag(v, a.top, 2) for v in tta_variants]),
                'bag_cent': merge_bags(
                    *[chain_bag(v, a.top) for v in tta_variants],
                    Counter(chunk_rank(pmodel, c['span'], feat, dev, cent_words,
                                       cent_mat, 5, a.chunk, a.stride))),
            }
            # перебор λ без повторного декода
            for lm in lam_grid:
                b = chain_bag([chains[rerank(chains, aff, default, lm)]])
                lam_totals[lm] += bag_hits(b, ref)
            row = {'id': c['id'], 'fold': k, 'phrase': c['phrase'], 'ref': ref,
                   'chains': chains[:a.keep], 'aff_top': sorted(
                       aff.items(), key=lambda kv: -kv[1])[:a.keep]}
            for v in variants:
                row[v] = list(bags[v].elements())
                row[v + '_hits'] = bag_hits(bags[v], ref)
                totals[v] += row[v + '_hits']
                pred_n[v] += sum(bags[v].values())
            rows.append(row)
        print(f'фолд {k} готов [{time.time() - t0:.0f} с]', flush=True)

    ref_total = sum(len(r['ref']) for r in rows)
    summary = {'clips': len(rows), 'ref_tokens': ref_total,
               'chains': a.nbest, 'beam': a.beam, 'word_bonus': a.word_bonus,
               'chunk': a.chunk, 'stride': a.stride, 'bag_top': a.top, 'tta': a.tta,
               'lam_rank': a.lam_rank, 'seconds': round(time.time() - t0, 1)}
    for v in variants:
        summary[v + '_hits'] = totals[v]
        summary[v + '_recall'] = round(totals[v] / max(ref_total, 1), 3)
        summary[v + '_precision'] = round(totals[v] / max(pred_n[v], 1), 3)
    summary['lam_curve'] = {str(lm): lam_totals[lm] for lm in lam_grid}
    out = {'tool': 'rerank_eval', 'date': time.strftime('%Y-%m-%d'),
           'model': Path(a.prod).name, 'summary': summary, 'rows': rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f'→ {a.out}', flush=True)


def selftest():
    ch = [['мать', 'я'], ['мать', 'мать', 'я'], ['нога']]
    assert chain_bag(ch, 2) == {'мать': 2, 'я': 1}  # max кратности, не сумма
    assert votes_bag(ch, 3, 2) == {'мать': 2, 'я': 2}
    assert votes_bag(ch, 3, 3) == {}
    assert bag_hits(chain_bag(ch, 2), ['мать', 'мать']) == 2
    assert bag_hits(chain_bag(ch, 2), ['нога']) == 0
    affs = {'мать': 0.9, 'нога': 0.8, 'я': 0.5}
    # без штрафа ранга выигрывает близость, с большим λ — CTC top-1
    assert rerank(ch, affs, 0.5, 0.0) == 2  # «нога» 0.8 > «мать+я» 0.7
    assert rerank(ch, affs, 0.5, 1.0) == 0
    # синонимы: берём max по частям, отсутствующие слова дают default
    assert word_affinity(['там/туда'], {'туда': 0.7}, 0.4) == 0.7
    assert word_affinity(['нет'], affs, 0.4) == 0.4
    assert merge_bags(Counter({'а': 2}), Counter({'а': 1, 'б': 1})) == {'а': 2, 'б': 1}
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='D:/hfdata/phrase_folds')
    ap.add_argument('--prod', default='D:/hfdata/phrase20/prod/prod.pt')
    ap.add_argument('--dict-data', default='datasets/youtube_dict/rutube_partial.npz')
    ap.add_argument('--inventory', default='reports/evidence/rutube_channel_inventory.json')
    ap.add_argument('--cache', default='D:/hfdata/rutube_phrases/landmarks')
    ap.add_argument('--out', default='reports/evidence/rerank_eval.json')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--nbest', type=int, default=20)
    ap.add_argument('--beam', type=int, default=24)
    ap.add_argument('--word-bonus', type=float, default=3.0)
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--keep', type=int, default=5)
    ap.add_argument('--lam', default='0,0.005,0.01,0.02,0.05,0.1')
    ap.add_argument('--lam-rank', type=float, default=0.01,
                    help='λ для варианта rerank; кривая по сетке --lam считается отдельно')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--canon', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--tta', default='1.0',
                    help='темпы TTA через запятую, например 1.0,0.85,1.2')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)


if __name__ == '__main__':
    main()