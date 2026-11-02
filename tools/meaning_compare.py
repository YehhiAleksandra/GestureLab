"""Смысл против слов: один и тот же честный прогон для нескольких конфигов декодера.

Свип (`sweep_decode`) считает попадание слов, но пользователь видит не слова, а перевод.
Здесь для каждого конфига тот же фолд-протокол (модель фолда не видела клип), те же клипы,
но дальше цепочки n-best уходят в LLM, а судья ставит 0/1/2 по смыслу. Парный бутстрап по
клипам показывает, различим ли перевод, а не только словарь.

  python tools/meaning_compare.py --inventory D:/hfdata/sp10_eval.json \\
      --cache D:/hfdata/sp10_rsl/landmarks --work D:/hfdata/dict_folds_honest
  python tools/meaning_compare.py --selftest

Канон по лемме включён по умолчанию: без него в оценку проходит вдвое меньше фраз.

Пустой top-1 в LLM не отправляется (B50) и засчитывается как смысл 0 без вызова.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dialogue_state import translate_in_context  # noqa: E402
from dialogue_translator import Recognizer  # noqa: E402
from eval_phrase_clips import words  # noqa: E402
from finetune_ctc_phrases import (build_canon, folds, head_names,  # noqa: E402
                                 load_phrase_clips, matched)
from finetune_ctc_signer import class_index  # noqa: E402
from gloss_to_text import SYSTEM, Translator, chrf, load_corpus  # noqa: E402
from sign_beam import GlossLM  # noqa: E402
from sweep_decode import corpus_seqs, phrase_seqs  # noqa: E402


def parse_decoders(spec):
    """'corpus:0.1:2.0:1,none:0.0:4.0:3' → конфиги; вес при lm_src=none обнуляется."""
    out = []
    for part in spec.split(','):
        f = part.strip().split(':')
        if len(f) != 4:
            sys.exit(f'конфиг «{part}» — ожидается lm_src:lm_weight:word_bonus:smooth')
        src, w, wb, sm = f[0], float(f[1]), float(f[2]), int(f[3])
        out.append({'lm_src': src, 'lm_weight': 0.0 if src == 'none' else w,
                    'word_bonus': wb, 'smooth': sm})
    return out


def boot_diff(a, b, n=4000, seed=2026):
    """Парный бутстрап разности средних по клипам (a, b — массивы одной длины)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    draws = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return {'mean_diff': round(float(d.mean()), 4),
            'lo95': round(float(np.percentile(draws, 2.5)), 4),
            'hi95': round(float(np.percentile(draws, 97.5)), 4),
            'a_wins': int((d > 0).sum()), 'b_wins': int((d < 0).sum()),
            'distinguishable': bool(np.percentile(draws, 2.5) > 0
                                   or np.percentile(draws, 97.5) < 0)}


def aggregate(rows, tr):
    n = max(len(rows), 1)
    judged = [r for r in rows if r.get('meaning') is not None]
    m = max(len(judged), 1)
    return {'clips': len(rows),
            'empty': round(sum(r['glosses'] == 0 for r in rows) / n, 3),
            'glosses_mean': round(sum(r['glosses'] for r in rows) / n, 2),
            'word_hits_canon': sum(r['hits_canon'] for r in rows),
            'word_tokens': sum(r['total'] for r in rows),
            'meaning_scored': len(judged),
            'llm_errors': sum(bool(r.get('error')) for r in rows),
            'meaning_ge1': round(sum(r['meaning'] >= 1 for r in judged) / m, 3),
            'meaning_2': round(sum(r['meaning'] == 2 for r in judged) / m, 3),
            'chrf': round(float(np.mean([r['chrf'] for r in judged])), 3) if judged else 0.0,
            'cost_usd': round(tr.cost, 4) if tr else 0.0}


def run(a):
    work = Path(a.work)
    ck = torch.load(work / 'fold0.pt', map_location='cpu', weights_only=False)
    names = list(ck['names'])
    index = class_index(names)
    inv = json.loads(Path(a.inventory).read_text(encoding='utf-8'))
    plain = [w for p in inv['phrases'] if p.get('duration') for w in words(p['phrase'])]
    canon, mode = build_canon(plain, head_names(names), a)
    clips, skipped = load_phrase_clips(a.cache, a.inventory, names, canon)
    if a.limit:
        clips = clips[:a.limit]
    cfgs = parse_decoders(a.decoder)
    print(f'канонизация ({mode}): клипов {len(clips)}, '
          f'пропущено {len(skipped)}; конфигураций {len(cfgs)}', flush=True)
    corpus = corpus_seqs(load_corpus('train'), index)
    tr = None if a.no_llm else Translator()
    parts = folds(len(clips), a.folds, a.seed)
    acc = {i: [] for i in range(len(cfgs))}
    t0 = time.time()
    for k, held_idx in enumerate(parts):
        held = [clips[i] for i in held_idx]
        train = [c for i, c in enumerate(clips) if i not in set(held_idx)]
        vocab = {w for c in train for w in c.get('canon_words', c['words'])}
        rec = Recognizer(str(work / f'fold{k}.pt'), decoder='beam', beam=24,
                         lm_weight=0.0, nbest=20)
        phr = phrase_seqs(train, index)
        lms = {'none': None, 'corpus': GlossLM(corpus, len(names)) if corpus else None,
               'phrases': GlossLM(phr, len(names)),
               'both': GlossLM(corpus + phr, len(names))}
        for ci, cfg in enumerate(cfgs):
            rec.lm = lms[cfg['lm_src']]
            rec.lm_weight, rec.word_bonus = cfg['lm_weight'], cfg['word_bonus']
            rec.smooth = cfg['smooth']
            for c in held:
                glosses, chains = rec.decode(c['span'])
                pred = [g for g, _ in glosses]
                pc = [canon.get(w, w) for w in pred]
                rc = ' '.join(canon.get(w, w) for w in c['words'])
                row = {'id': c['id'], 'fold': k, 'phrase': c['phrase'],
                       'pred': pred, 'glosses': len(pred),
                       'hits_canon': sum(matched(pc, rc).values()),
                       'total': len(c['words']),
                       'seen_hits': sum(v for w, v in matched(pred, c['phrase']).items()
                                        if w in vocab),
                       'seen_total': sum(w in vocab for w in
                                         c.get('canon_words', c['words'])),
                       'meaning': None, 'chrf': None}
                text = ''
                if tr and pred:
                    try:
                        text = translate_in_context(tr, glosses, [], SYSTEM, chains=chains)
                        row['meaning'] = tr.judge(text, c['phrase']) if text else 0
                        row['chrf'] = round(chrf(text, c['phrase']), 3)
                    except Exception as e:  # noqa: BLE001 — один сбой сети не роняет прогон
                        row['error'] = f'{type(e).__name__}: {e}'
                elif tr:
                    # B50: пустой top-1 не судим — это 0 по смыслу, а не пропуск
                    row['meaning'] = 0
                    row['chrf'] = round(chrf('', c['phrase']), 3)
                row['translation'] = text
                acc[ci].append(row)
        print(f'фолд {k} готов [{time.time() - t0:.0f} с, ${tr.cost if tr else 0:.4f}]',
              flush=True)

    base = cfgs[0]
    results = []
    for ci, cfg in enumerate(cfgs):
        rows = acc[ci]
        r = {**cfg, 'config_index': ci, **aggregate(rows, tr)}
        if ci:
            # списки построены в одном порядке (фолд → клип), поэтому пары выровнены
            r['vs_base'] = {
                'meaning': boot_diff([x['meaning'] or 0 for x in rows],
                                     [x['meaning'] or 0 for x in acc[0]]),
                'hits_canon': boot_diff([x['hits_canon'] for x in rows],
                                        [x['hits_canon'] for x in acc[0]]),
                'chrf': boot_diff([x['chrf'] or 0.0 for x in rows],
                                  [x['chrf'] or 0.0 for x in acc[0]])}
        results.append(r)
    for r in results:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    out = {'tool': 'meaning_compare', 'date': time.strftime('%Y-%m-%d'),
           'inventory': str(a.inventory), 'work': str(work), 'folds': a.folds,
           'seed': a.seed, 'canon': mode,
           'base': base, 'seconds': round(time.time() - t0, 1), 'results': results,
           'rows': {str(ci): acc[ci] for ci in range(len(cfgs))}}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                               encoding='utf-8')
        print(f'→ {a.out}', flush=True)
    return out


def selftest():
    cfgs = parse_decoders('corpus:0.1:2.0:1,none:0.05:4.0:3')
    assert cfgs[1]['lm_weight'] == 0.0, 'без LM вес не значит ничего'
    assert cfgs[0]['lm_weight'] == 0.1 and cfgs[1]['smooth'] == 3, cfgs
    d = boot_diff([2, 2, 2, 2, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0])
    assert d['mean_diff'] == 0.75 and d['a_wins'] == 4 and d['b_wins'] == 0, d
    same = boot_diff([1, 1, 0, 0], [1, 1, 0, 0])
    assert same['mean_diff'] == 0.0 and not same['distinguishable'], same
    rows = [{'glosses': 0, 'hits_canon': 0, 'total': 2, 'meaning': 0, 'chrf': 0.0},
            {'glosses': 3, 'hits_canon': 1, 'total': 2, 'meaning': 2, 'chrf': 40.0}]
    s = aggregate(rows, None)
    assert s['empty'] == 0.5 and s['meaning_ge1'] == 0.5 and s['meaning_2'] == 0.5, s
    assert s['word_hits_canon'] == 1 and s['word_tokens'] == 4, s
    print('selftest ok')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='D:/hfdata/dict_folds_honest')
    ap.add_argument('--cache', default='D:/hfdata/sp10_rsl/landmarks')
    ap.add_argument('--inventory', default='D:/hfdata/sp10_eval.json')
    ap.add_argument('--decoder',
                    default='corpus:0.1:2.0:1,none:0.0:4.0:3',
                    help='первый конфиг — база сравнения')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--canon', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--lemma', action=argparse.BooleanOptionalAction, default=True,
                    help='канон по лемме (pymorphy3); без пакета — автоматом по основе')
    ap.add_argument('--no-llm', action='store_true')
    ap.add_argument('--out', default='reports/evidence/meaning_compare.json')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(a)


if __name__ == '__main__':
    main()
