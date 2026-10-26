"""Замер перевода без распознавания: правила РЖЯ в промпте и выбор модели.

Вход — сохранённые строки `eval_lattice` (глоссы beam, решётка n-best, эталон), поэтому
GPU и CTC не нужны: меняется только текст промпта и модель. Судья тот же, что в этапе 9,
шкала 0/1/2; chrF считается по переводу.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from math import erfc, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gloss_to_text import LATTICE, SYSTEM, SYSTEM_RULES, Translator, chrf

PROMPTS = {
    'base': (SYSTEM, False, None),
    'lattice': (SYSTEM + LATTICE, True, None),
    # контроль: промпт про варианты без самих вариантов — вклад дают кандидаты или текст?
    'lattice_nocands': (SYSTEM + LATTICE, False, None),
    'rules': (SYSTEM_RULES, False, None),
    'rules_noshot': (SYSTEM_RULES, False, 0),
    'rules_lattice': (SYSTEM_RULES + LATTICE, True, None),
}


def load_rows(paths):
    seen, rows = set(), []
    for path in paths:
        for r in json.loads(Path(path).read_text(encoding='utf-8'))['rows']:
            key = (r['glosses_ref'], r['ref'])
            if key in seen:
                continue
            seen.add(key)
            if not r.get('cands'):
                raise ValueError(f'{path}: у строки нет candidates — нужен eval_lattice --no-llm')
            rows.append(r)
    return rows


def bracketed(cands):
    return ' '.join('[' + ' | '.join(c) + ']' for c in cands)


def flat(cands):
    return ' '.join(w for c in cands for w in c)


def translate(tr, row, variant):
    system, use_lattice, shots = PROMPTS[variant]
    if use_lattice:
        return tr(flat(row['cands']), system, bracketed(row['cands']))
    if shots == 0:
        return tr.chat([{'role': 'system', 'content': system},
                        {'role': 'user', 'content': row['glosses_beam']}], 80)
    return tr(row['glosses_beam'], system)


def run_variant(tr, rows, variant, workers):
    def one(r):
        try:
            text = translate(tr, r, variant)
            return {'hyp': text, 'judge': tr.judge(text, r['ref'])}
        except Exception as e:  # один вызов без ретраев: ошибка идёт в отчёт
            return {'error': str(e)}
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, rows))
    return res, round(time.time() - t0)


def summarize(res, refs):
    ok = [r['judge'] for r in res if isinstance(r.get('judge'), int)]
    ch = [chrf(r['hyp'], ref) for r, ref in zip(res, refs)
          if isinstance(r.get('hyp'), str) and r['hyp']]
    return {'meaning_ok': round(sum(v == 2 for v in ok) / max(len(ok), 1), 3),
            'meaning_gist': round(sum(v >= 1 for v in ok) / max(len(ok), 1), 3),
            'meaning_mean': round(float(np.mean(ok)), 3) if ok else None,
            'chrf': round(float(np.mean(ch)), 1) if ch else None,
            'judged': len(ok), 'errors': sum('error' in r for r in res)}


def paired(a, b, n=10000, seed=9):
    diffs = np.array([x['judge'] - y['judge'] for x, y in zip(a, b)
                      if isinstance(x.get('judge'), int) and isinstance(y.get('judge'), int)])
    if not len(diffs):
        return None
    rng = np.random.default_rng(seed)
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n)])
    wins, losses = int((diffs > 0).sum()), int((diffs < 0).sum())
    z = (wins - (wins + losses) / 2) / sqrt((wins + losses) / 4) if wins + losses else 0.0
    return {'wins': wins, 'losses': losses, 'ties': int((diffs == 0).sum()),
            'mean_diff': round(float(diffs.mean()), 3),
            'mean_diff_ci95': [round(float(np.percentile(boot, 2.5)), 3),
                               round(float(np.percentile(boot, 97.5)), 3)],
            'sign_test_p': round(float(erfc(abs(z) / sqrt(2))), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', nargs='+',
                    default=['reports/evidence/lattice_rows_111.json',
                             'reports/evidence/lattice_rows_130.json'])
    ap.add_argument('--variants', default='base,rules,rules_noshot,rules_lattice')
    ap.add_argument('--model', default='google/gemma-4-31B-it')
    ap.add_argument('--alt-models',
                    default='meta-llama/Llama-3.3-70B-Instruct,deepseek-ai/DeepSeek-V3')
    ap.add_argument('--alt-variant', default='rules')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='reports/evidence/translation_eval.json')
    a = ap.parse_args()

    rows = load_rows(a.rows)
    refs = [r['ref'] for r in rows]
    print(f'фраз {len(rows)}', flush=True)
    out = {'phrases': len(rows), 'rows': a.rows, 'model': a.model,
           'alt_models': [m for m in a.alt_models.split(',') if m],
           'results': {}, 'cost_usd': 0.0}
    runs, seconds = {}, {}

    tr = Translator(a.model)
    for variant in [v for v in a.variants.split(',') if v]:
        res, sec = run_variant(tr, rows, variant, a.workers)
        runs[variant], seconds[variant] = res, sec
        sc = summarize(res, refs)
        out['results'][variant] = {**sc, 'seconds': sec}
        print(variant, sc, f'{sec} с', flush=True)
    out['cost_usd'] += tr.cost

    for model in out['alt_models']:
        label = f'{a.alt_variant}@{model.split("/")[-1]}'
        alt = Translator(model)
        res, sec = run_variant(alt, rows, a.alt_variant, a.workers)
        runs[label], seconds[label] = res, sec
        sc = summarize(res, refs)
        out['results'][label] = {**sc, 'seconds': sec}
        print(label, sc, f'{sec} с', flush=True)
        out['cost_usd'] += alt.cost

    base = runs.get('base')
    if base:
        out['vs_base'] = {name: paired(res, base) for name, res in runs.items()
                          if name != 'base'}
    rules = runs.get(a.alt_variant)
    if rules:
        out['vs_rules'] = {name: paired(res, rules) for name, res in runs.items()
                           if name != a.alt_variant}
    out['cost_usd'] = round(out['cost_usd'], 4)
    out['rows_out'] = {name: [{'hyp': r.get('hyp'), 'judge': r.get('judge'),
                               'error': r.get('error')} for r in res]
                       for name, res in runs.items()}
    print(json.dumps({k: v for k, v in out.items() if k != 'rows_out'},
                     ensure_ascii=False, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
