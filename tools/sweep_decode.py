"""Перебор декодера на готовых фолдах этапа 19 — без обучения, только декод.

Фолд-модель не видела отложенные клипы; LM собирается из корпуса RSL и, в варианте
`both`, из слов фраз обучающих клипов этого фолда. Фраза отложенного клипа в LM не
попадает, поэтому сравнение остаётся честным.

  python tools/sweep_decode.py --quick
  python tools/sweep_decode.py --lm-weight 0.1,0.25 --beam 24,48 --smooth 1,3
  python tools/sweep_decode.py --selftest
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dialogue_translator import Recognizer  # noqa: E402
from eval_phrase_clips import hits, words  # noqa: E402
from finetune_ctc_phrases import canon_map, folds, load_phrase_clips, matched  # noqa: E402
from gloss_to_text import load_corpus  # noqa: E402
from expand_ctc_head import expand_checkpoint  # noqa: E402
from sign_beam import GlossLM  # noqa: E402


def configs(lm_src, lm_weight, word_bonus, smooth, beam):
    """Декартово произведение, без бессмысленных дублей: без LM вес не значит ничего."""
    out, seen = [], set()
    for s, w, b, sm, bm in product(lm_src, lm_weight, word_bonus, smooth, beam):
        w = 0.0 if s == 'none' else float(w)
        key = (s, w, float(b), int(sm), int(bm))
        if key in seen:
            continue
        seen.add(key)
        out.append({'lm_src': s, 'lm_weight': w, 'word_bonus': float(b),
                    'smooth': int(sm), 'beam': int(bm)})
    return out


def corpus_seqs(corpus, index):
    """Фразы корпуса, целиком покрытые классами головы, → id-последовательности."""
    return [[index[w.lower()] for w in p['rsl'].split()] for p in corpus
            if p['rsl'].split() and all(w.lower() in index for w in p['rsl'].split())]


def phrase_seqs(clips, index):
    return [[index[w.lower()] for w in c.get('canon_words', c['words'])] for c in clips
            if all(w.lower() in index for w in c.get('canon_words', c['words']))]


def summarize_rows(rows):
    n = max(len(rows), 1)
    hits_n = sum(r['hits'] for r in rows)
    canon_n = sum(r['hits_canon'] for r in rows)
    tokens = sum(r['total'] for r in rows)
    pred_n = sum(r['glosses'] for r in rows)
    seen_hits = sum(r['seen_hits'] for r in rows)
    seen_total = sum(r['seen_total'] for r in rows)
    return {'clips': len(rows),
            'empty': round(sum(r['glosses'] == 0 for r in rows) / n, 3),
            'glosses_mean': round(pred_n / n, 2),
            'word_hits': hits_n, 'word_tokens': tokens,
            'word_recall': round(hits_n / max(tokens, 1), 3),
            'word_precision': round(hits_n / max(pred_n, 1), 3),
            'word_hits_canon': canon_n,
            'word_recall_canon': round(canon_n / max(tokens, 1), 3),
            'word_precision_canon': round(canon_n / max(pred_n, 1), 3),
            'phrase_all_words': sum(r['hits'] == r['total'] > 0 for r in rows),
            'phrase_all_words_canon': sum(r['hits_canon'] == r['total'] > 0 for r in rows),
            'seen_hits': seen_hits, 'seen_tokens': seen_total,
            'seen_recall': round(seen_hits / max(seen_total, 1), 3)}


def run(a):
    work = Path(a.work)
    ck = torch.load(work / 'fold0.pt', map_location='cpu', weights_only=False)
    names = list(ck['names'])
    index = {n.lower(): i for i, n in enumerate(names)}

    vocab_all = {w.lower() for w in names}
    inv = json.loads(Path(a.inventory).read_text(encoding='utf-8'))
    plain = [w for p in inv['phrases'] if p.get('duration') for w in words(p['phrase'])]
    canon = canon_map(plain, vocab_all) if a.canon else {}
    clips, skipped = load_phrase_clips(a.cache, a.inventory, names, canon)
    print(f'канонизация: {sum(1 for w in plain if canon.get(w, w) != w)} слов; '
          f'клипов {len(clips)}, пропущено {len(skipped)}', flush=True)
    corpus = corpus_seqs(load_corpus('train'), index)
    print(f'корпус {len(corpus)} фраз', flush=True)

    parts = folds(len(clips), a.folds, a.seed)
    cfgs = configs([s.strip() for s in a.lm_src.split(',')],
                   [float(x) for x in a.lm_weight.split(',')],
                   [float(x) for x in a.word_bonus.split(',')],
                   [int(x) for x in a.smooth.split(',')],
                   [int(x) for x in a.beam.split(',')])
    if a.quick:
        cfgs = cfgs[:2]
    print(f'конфигураций {len(cfgs)}', flush=True)

    acc = {i: [] for i in range(len(cfgs))}
    t0 = time.time()
    for k, held_idx in enumerate(parts):
        if a.quick and k > 0:
            break
        held = [clips[i] for i in held_idx]
        train = [c for i, c in enumerate(clips) if i not in set(held_idx)]
        vocab = {w for c in train for w in c.get('canon_words', c['words'])}
        rec = Recognizer(str(work / f'fold{k}.pt'), decoder='beam', lm_weight=0.0)
        phr = phrase_seqs(train, index)
        lms = {'none': None,
               'corpus': GlossLM(corpus, len(names)) if corpus else None,
               # слова фраз редки в корпусе RSL → корпусная LM их штрафует; домен отдельно
               'phrases': GlossLM(phr, len(names)),
               'both': GlossLM(corpus + phr, len(names))}
        for ci, cfg in enumerate(cfgs):
            rec.lm = lms[cfg['lm_src']]
            rec.lm_weight, rec.word_bonus = cfg['lm_weight'], cfg['word_bonus']
            rec.smooth, rec.beam = cfg['smooth'], cfg['beam']
            for c in held:
                pred = [g for g, _ in rec.decode(c['span'])[0]]
                m = matched(pred, c['phrase'])
                pc = [canon.get(w, w) for w in pred]
                rc = ' '.join(canon.get(w, w) for w in c['words'])
                acc[ci].append({'id': c['id'], 'fold': k, 'pred': pred,
                                'glosses': len(pred), 'hits': hits(pred, c['phrase'])[0],
                                'hits_canon': sum(matched(pc, rc).values()),
                                'total': len(c['words']),
                                'seen_hits': sum(v for w, v in m.items() if w in vocab),
                                'seen_total': sum(w in vocab for w in
                                                  c.get('canon_words', c['words']))})
        print(f'фолд {k} готов [{time.time() - t0:.0f} с]', flush=True)

    results = []
    for ci, cfg in enumerate(cfgs):
        results.append({**cfg, 'config_index': ci, **summarize_rows(acc[ci])})
    results.sort(key=lambda r: (-r['word_hits'], r['empty']))
    for r in results[:10]:
        print(json.dumps(r, ensure_ascii=False), flush=True)

    out = {'tool': 'sweep_decode', 'date': time.strftime('%Y-%m-%d'),
           'folds': a.folds, 'seed': a.seed, 'clips': len(clips),
           'config': {'lm_src': a.lm_src, 'lm_weight': a.lm_weight,
                      'word_bonus': a.word_bonus, 'smooth': a.smooth, 'beam': a.beam},
           'seconds': round(time.time() - t0, 1), 'results': results,
           'rows': {str(ci): acc[ci] for ci in range(len(cfgs))}}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                               encoding='utf-8')
        print(f'→ {a.out}', flush=True)
    return out


def selftest():
    cf = configs(['none', 'corpus'], [0.0, 0.1], [2.0], [1], [24])
    assert len(cf) == 3, cf  # none×вес схлопывается, corpus×0 и corpus×0.1 разные
    assert all(c['lm_weight'] == 0.0 for c in cf if c['lm_src'] == 'none')
    rows = [{'glosses': 0, 'hits': 0, 'total': 2, 'hits_canon': 0,
             'seen_hits': 0, 'seen_total': 1},
            {'glosses': 4, 'hits': 1, 'total': 3, 'hits_canon': 2,
             'seen_hits': 1, 'seen_total': 2}]
    s = summarize_rows(rows)
    assert s['empty'] == 0.5 and s['word_hits'] == 1 and s['word_tokens'] == 5
    assert s['word_precision'] == 0.25 and s['phrase_all_words'] == 0
    assert s['word_hits_canon'] == 2 and s['word_recall_canon'] == 0.4
    assert s['seen_recall'] == round(1 / 3, 3)
    idx = {'я': 0, 'мыть': 1}
    assert corpus_seqs([{'rsl': 'я мыть'}, {'rsl': 'я незнакомое'}], idx) == [[0, 1]]
    assert phrase_seqs([{'words': ['мыть', 'я']}], idx) == [[1, 0]]
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='D:/hfdata/phrase_folds')
    ap.add_argument('--init', default='models/slovo_ctc_motion_head_all.pt')
    ap.add_argument('--cache', default='D:/hfdata/rutube_phrases/landmarks')
    ap.add_argument('--inventory', default='reports/evidence/rutube_channel_inventory.json')
    ap.add_argument('--out', default='reports/evidence/decode_sweep.json')
    ap.add_argument('--lm-src', default='none,corpus,both')
    ap.add_argument('--lm-weight', default='0.1,0.25,0.5')
    ap.add_argument('--word-bonus', default='0,2,3')
    ap.add_argument('--smooth', default='1')
    ap.add_argument('--beam', default='24')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--canon', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(a)


if __name__ == '__main__':
    main()
