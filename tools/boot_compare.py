"""Парный бутстрап по клипам: различим ли выбор декодера вообще.

Свип подбирает конфиг по числу угаданных слов на десятках клипов; разница между
лучшим и baseline там обычно 1–4 слова. Эта скрипка считает разницу по каждому клипу
и бутстрапит её по клипам, то есть отвечает не «кто первый в списке», а «отличимы ли
они от шума».

  python tools/boot_compare.py --sweep D:/hfdata/sp10_tune_sweep.json --a 11 --b 13
  python tools/boot_compare.py --selftest
"""
import argparse
import json
from pathlib import Path

import numpy as np


def per_clip(rows, key='hits_canon'):
    out = {}
    for r in rows:
        out[(r['fold'], r['id'])] = out.get((r['fold'], r['id']), 0) + int(r[key])
    return out


def paired_bootstrap(a, b, n=4000, seed=2026):
    """a, b — словари клип→счёт. Возвращает точку оценки и 95% ЦИ разницы (a - b)."""
    keys = sorted(set(a) & set(b))
    va = np.array([a[k] for k in keys], np.float64)
    vb = np.array([b[k] for k in keys], np.float64)
    d = va - vb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(1)
    return {'clips': len(keys), 'mean_diff': round(float(d.mean()), 4),
            'lo95': round(float(np.quantile(boot, 0.025)), 4),
            'hi95': round(float(np.quantile(boot, 0.975)), 4),
            'p_a_better': round(float((boot > 0).mean()), 4),
            'a_wins': int((d > 0).sum()), 'b_wins': int((d < 0).sum()),
            'ties': int((d == 0).sum())}


def describe(cfg):
    return (f"{cfg['lm_src']}/w{cfg['lm_weight']}/wb{cfg['word_bonus']}"
            f"/sm{cfg['smooth']}/bm{cfg['beam']}")


def run(a):
    sw = json.loads(Path(a.sweep).read_text(encoding='utf-8'))
    by_i = {r['config_index']: r for r in sw['results']}
    rows = {int(k): v for k, v in sw['rows'].items()}
    ca, cb = by_i[a.a], by_i[a.b]
    res = paired_bootstrap(per_clip(rows[a.a]), per_clip(rows[a.b]))
    out = {'sweep': a.sweep, 'metric': a.metric,
           'a': {**{k: ca[k] for k in ('lm_src', 'lm_weight', 'word_bonus', 'smooth',
                                       'beam')},
                 'word_recall_canon': ca['word_recall_canon'],
                 'word_precision_canon': ca['word_precision_canon'],
                 'word_hits_canon': ca['word_hits_canon']},
           'b': {**{k: cb[k] for k in ('lm_src', 'lm_weight', 'word_bonus', 'smooth',
                                       'beam')},
                 'word_recall_canon': cb['word_recall_canon'],
                 'word_precision_canon': cb['word_precision_canon'],
                 'word_hits_canon': cb['word_hits_canon']},
           'a_desc': describe(ca), 'b_desc': describe(cb), **res}
    ci = 0 if res['lo95'] <= 0 <= res['hi95'] else 1
    print(json.dumps(out, ensure_ascii=False), flush=True)
    print(f'различимы: {bool(ci)} (ЦИ [{res["lo95"]}, {res["hi95"]}] '
          f'{"содержит" if not ci else "не содержит"} 0)', flush=True)
    return out


def selftest():
    a = {(0, 'x'): 3, (0, 'y'): 0, (1, 'z'): 1}
    b = {(0, 'x'): 2, (0, 'y'): 1, (1, 'z'): 1, (1, 'w'): 5}
    r = paired_bootstrap(a, b, n=200)
    assert r['clips'] == 3, r          # общий ключ, 'w' не参与了 сравнение
    assert r['mean_diff'] == 0.0 and r['ties'] == 1  # +1, -1, 0 по клипам
    assert 0 <= r['p_a_better'] <= 1 and r['lo95'] <= r['hi95']
    assert per_clip([{'fold': 0, 'id': 'x', 'hits_canon': 2},
                     {'fold': 0, 'id': 'x', 'hits_canon': 1}]) == {(0, 'x'): 3}
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default='D:/hfdata/sp10_tune_sweep.json')
    ap.add_argument('--a', type=int, required=False)
    ap.add_argument('--b', type=int, required=False)
    ap.add_argument('--metric', default='hits_canon')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.a is None or a.b is None:
        ap.error('--a и --b (config_index) обязательны')
    run(a)


if __name__ == '__main__':
    main()
