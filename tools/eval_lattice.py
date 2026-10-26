"""Сквозная оценка перевода с CTC-потоков: greedy vs beam+LM vs n-best lattice.

Один и тот же поток на фразу, логиты считаются один раз. Сравниваются четыре входа LLM:
greedy top-1, beam+LM top-1, решётка из n-best и эталонные глоссы фразы (потолок перевода).
Понятность меряет судья 0/1/2, качество текста — chrF.

Тестовые фразы — ровно те же отложенные 51, что в tools/eval_decoders.py: LM училась
на 80% уникальных фраз корпуса, эти цепочки в неё не входили.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gloss_to_text import Translator, chrf, load_corpus
from sign_beam import GlossLM, ctc_beam_search, nbest_to_lattice
from sign_stream import load_ctc
from train_ctc_slovo import (
    batch_features,
    corpus_phrases,
    edit_counts,
    make_corpus_stream,
    prepare_slovo,
)


def heldout_phrases(names, d, dev_size=60, holdout_seed=2027):
    """Те же отложенные фразы и тот же порядок, что в eval_decoders (seed 2027/2026)."""
    index = {n.lower(): i for i, n in enumerate(names)}
    train = corpus_phrases(index, 'train')
    test = []
    for p in load_corpus('test'):
        ids = [index.get(w.lower()) for w in p['rsl'].split()]
        if ids and all(i is not None for i in ids) and all(i in d['te_by_class'] for i in ids):
            test.append({'ids': ids, 'rsl': ' '.join(p['rsl'].split()),
                         'russian': p['russian']})
    uniq = sorted({tuple(p) for p in train}, key=lambda t: (len(t), t))
    perm = np.random.default_rng(holdout_seed).permutation(len(uniq))
    held = {uniq[i] for i in perm[:len(uniq) // 5]}
    lm_seqs = [list(uniq[i]) for i in perm[len(uniq) // 5:]]
    phrases = [p for p in test if tuple(p['ids']) in held]
    order = np.random.default_rng(2026).permutation(len(phrases))
    phrases = [phrases[i] for i in order]
    return lm_seqs, phrases[dev_size:]


def greedy(logp):
    ids, out, prev = logp.argmax(-1).tolist(), [], None
    for i in ids:
        if i != prev and i != 0:
            out.append(i - 1)
        prev = i
    return out


@torch.no_grad()
def logits_of(model, stream, dev):
    x, lens = batch_features([stream])
    return model(x.to(dev), lens.to(dev))[0].cpu().log_softmax(-1).numpy()


def bootstrap_ci(diffs, n=2000, seed=0):
    if not diffs:
        return None, None
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(diffs, len(diffs), replace=True))) for _ in range(n)]
    return round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/slovo/slovo_holistic.npz')
    ap.add_argument('--model', default='models/slovo_ctc_corpus.pt')
    ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--nbest', type=int, default=5)
    ap.add_argument('--max-cands', type=int, default=4)
    ap.add_argument('--lm-weight', type=float, default=0.1)
    ap.add_argument('--word-bonus', type=float, default=2.0)
    ap.add_argument('--llm', default='google/gemma-4-31B-it')
    ap.add_argument('--stream-seed', type=int, default=4242)
    ap.add_argument('--dev-size', type=int, default=60,
                    help='сколько отложенных фраз отвести под подбор (0 — оценивать все)')
    ap.add_argument('--holdout-seed', type=int, default=2027,
                    help='сид разбиения уникальных фраз: другой сид — другой независимый набор')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-llm', action='store_true',
                    help='только распознавание и кандидаты, без переводчика и судьи')
    ap.add_argument('--out', default='reports/evidence/lattice_eval.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = prepare_slovo(a.data)
    model, names, _ = load_ctc(a.model, dev)
    lm_seqs, phrases = heldout_phrases(names, d, dev_size=a.dev_size,
                                       holdout_seed=a.holdout_seed)
    lm = GlossLM(lm_seqs, n_classes=len(names))
    rng = np.random.default_rng(a.stream_seed)
    print(f'модель {Path(a.model).name}; фраз {len(phrases)}; LM по {len(lm_seqs)} фразам',
          flush=True)

    rows = []
    for phrase in phrases:
        stream, labels = make_corpus_stream(d['te_by_class'], [phrase['ids']], rng,
                                            augment_clips=False, pause_prob=0.0, blend_prob=1.0)
        logp = logits_of(model, stream, dev)
        hyps = ctc_beam_search(logp, beam=a.beam, topk=a.topk, nbest=a.nbest, lm=lm,
                              lm_weight=a.lm_weight, word_bonus=a.word_bonus)
        cands = nbest_to_lattice([h for h, _ in hyps], a.max_cands)
        rows.append({'ref': phrase['russian'], 'oracle_text': phrase['rsl'], 'labels': labels,
                     'greedy': greedy(logp), 'beam': hyps[0][0],
                     'nbest': [h for h, _ in hyps], 'cands': cands})

    def rec_acc(key):
        pos = sum(sum(x == y for x, y in zip(r[key], r['labels'])) for r in rows)
        total = sum(len(r['labels']) for r in rows)
        exact = sum(r[key] == r['labels'] for r in rows) / len(rows)
        edit = sum(edit_counts(r[key], r['labels']) for r in rows) / total
        return {'gloss_acc': round(pos / total, 4), 'exact_acc': round(exact, 4),
                'edit_rate': round(edit, 4)}

    if a.no_llm:
        res_rows = [{'ref': r['ref'], 'glosses_ref': r['oracle_text'],
                     'glosses_greedy': ' '.join(names[i] for i in r['greedy']),
                     'glosses_beam': ' '.join(names[i] for i in r['beam']),
                     'cands': [[names[i] for i in c] for c in r['cands']],
                     'oracle_exact_nbest': int(r['labels'] in r['nbest'])} for r in rows]
        summary = {'model': a.model, 'phrases': len(rows), 'beam': a.beam,
                   'nbest': a.nbest, 'lm_weight': a.lm_weight,
                   'word_bonus': a.word_bonus, 'holdout_seed': a.holdout_seed,
                   'dev_size': a.dev_size,
                   'recognition': {'greedy': rec_acc('greedy'), 'beam': rec_acc('beam')},
                   'nbest_oracle_exact': round(sum(r['oracle_exact_nbest'] for r in res_rows)
                                               / len(res_rows), 4),
                   'llm': None}
        print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({**summary, 'rows': res_rows}, ensure_ascii=False,
                                         indent=1), encoding='utf-8')
        return

    tr = Translator(a.llm)
    t0 = time.time()

    def one(r):
        g = ' '.join(names[i] for i in r['greedy'])
        b = ' '.join(names[i] for i in r['beam'])
        lt = [[names[i] for i in c] for c in r['cands']]
        try:
            out = {'greedy': tr(g), 'beam': tr(b), 'lattice': tr.lattice(lt),
                   'oracle': tr(r['oracle_text'])}
        except Exception as e:  # один вызов без ретраев: ошибка идёт в отчёт
            return {'error': str(e)}
        try:
            out.update({f'meaning_{k}': tr.judge(v, r['ref'])
                        for k, v in out.items()})
        except Exception as e:
            out['judge_error'] = str(e)
        return out

    with ThreadPoolExecutor(a.workers) as ex:
        hyps = list(ex.map(one, rows))

    modes = ['oracle', 'beam', 'lattice', 'greedy']
    res_rows = []
    for r, h in zip(rows, hyps):
        row = {'ref': r['ref'], 'glosses_ref': r['oracle_text'],
               'glosses_greedy': ' '.join(names[i] for i in r['greedy']),
               'glosses_beam': ' '.join(names[i] for i in r['beam']),
               'cands': [[names[i] for i in c] for c in r['cands']],
               'oracle_exact_nbest': int(r['labels'] in r['nbest']), **h}
        res_rows.append(row)

    def scores(mode):
        return [r[f'meaning_{mode}'] for r in res_rows if r.get(f'meaning_{mode}') is not None]

    summary = {'model': a.model, 'llm': a.llm, 'phrases': len(rows),
               'beam': a.beam, 'nbest': a.nbest, 'lm_weight': a.lm_weight,
               'word_bonus': a.word_bonus, 'holdout_seed': a.holdout_seed,
               'dev_size': a.dev_size,
               'recognition': {'greedy': rec_acc('greedy'), 'beam': rec_acc('beam')},
               'nbest_oracle_exact': round(sum(r['oracle_exact_nbest'] for r in res_rows)
                                           / len(res_rows), 4),
               'llm_errors': sum('error' in r for r in res_rows),
               'judge_missing': sum(r.get(f'meaning_{m}') is None
                                    for r in res_rows for m in modes),
               'cost_usd': round(tr.cost, 4), 'seconds': round(time.time() - t0)}
    for m in modes:
        s = scores(m)
        summary[f'meaning_ok_{m}'] = round(sum(v >= 2 for v in s) / max(len(s), 1), 4)
        summary[f'meaning_gist_{m}'] = round(sum(v >= 1 for v in s) / max(len(s), 1), 4)
        summary[f'meaning_mean_{m}'] = round(float(np.mean(s)), 3) if s else None
        ch = [chrf(r[m], r['ref']) for r in res_rows if isinstance(r.get(m), str)]
        summary[f'chrf_{m}'] = round(float(np.mean(ch)), 1) if ch else None

    pairs = [(r['meaning_lattice'], r['meaning_beam']) for r in res_rows
             if r.get('meaning_lattice') is not None and r.get('meaning_beam') is not None]
    diffs = [x - y for x, y in pairs]
    summary['lattice_vs_beam'] = {
        'wins': sum(d > 0 for d in diffs), 'losses': sum(d < 0 for d in diffs),
        'ties': sum(d == 0 for d in diffs),
        'mean_diff': round(float(np.mean(diffs)), 3) if diffs else None,
        'mean_diff_ci95': bootstrap_ci(diffs)}

    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({**summary, 'rows': res_rows}, ensure_ascii=False,
                                     indent=1), encoding='utf-8')
    for r in res_rows[:6]:
        if 'error' in r:
            continue
        print(f"\n  эталон: {r['glosses_ref']}\n  beam:   {r['glosses_beam']}"
              f"\n  beam →  {r['beam']}  [{r['meaning_beam']}]"
              f"\n  lattice → {r['lattice']}  [{r['meaning_lattice']}]", flush=True)


if __name__ == '__main__':
    main()
