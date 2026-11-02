"""Гейт утечки для сплитов: train-инвентарь и оценочный не должны быть одной записью.

check_dict_disjoint сверяет словарь канала с фразовыми клипами. Здесь пара теснее: tune и
dev+test одного бенчмарка лежат в одном каталоге ландмарок и различаются только id, поэтому
проверка по именам ничего не стоит — один и тот же жест мог попасть в оба списка под
разными id. Сравниваются отпечатки окон кадров, как в гейте словаря.

  python tools/check_split_disjoint.py --a D:/hfdata/sp10_tune.json \\
      --b D:/hfdata/sp10_eval.json --cache D:/hfdata/sp10_rsl/landmarks \\
      --out reports/evidence/sp10_tune_eval_disjoint.json
  python tools/check_split_disjoint.py --selftest

Код возврата 0 — сплиты не пересекаются, 1 — утечка.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

from check_dict_disjoint import _h  # noqa: E402


def ids(inventory):
    """Клипы инвентаря с длительностью — ровно те, что берутся в обучение и в оценку."""
    inv = json.loads(Path(inventory).read_text(encoding='utf-8'))
    return {p['id'] for p in inv['phrases'] if p.get('duration')}


def windows(idset, cache, window=4):
    """{отпечаток окна: id} по клипам кэша; недостающие npz просто не участвуют."""
    out = {}
    missing = 0
    for i in sorted(idset):
        p = Path(cache) / f'{i}.npz'
        if not p.exists():
            missing += 1
            continue
        with np.load(p) as d:
            f = np.ascontiguousarray(d['frames'])
        for k in range(len(f) - window + 1):
            out.setdefault(_h(f[k:k + window]), i)
    return out, missing


def run(a):
    ia, ib = ids(a.a), ids(a.b)
    wa, miss_a = windows(ia, a.cache, a.window)
    wb, miss_b = windows(ib, a.cache, a.window)
    shared = sorted(set(wa) & set(wb))
    res = {'tool': 'check_split_disjoint', 'date': time.strftime('%Y-%m-%d'),
           'a': str(a.a), 'b': str(a.b), 'cache': str(a.cache), 'window': a.window,
           'ids_a': len(ia), 'ids_b': len(ib), 'ids_shared': len(ia & ib),
           'windows_a': len(wa), 'windows_b': len(wb), 'windows_shared': len(shared),
           'no_cache_a': miss_a, 'no_cache_b': miss_b,
           'examples': [{'a': wa[s], 'b': wb[s]} for s in shared[:20]],
           'disjoint': not (ia & ib) and not shared}
    print(json.dumps({k: res[k] for k in ('ids_a', 'ids_b', 'ids_shared', 'windows_a',
                                          'windows_b', 'windows_shared', 'disjoint')},
                     ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n',
                               encoding='utf-8')
        print(f'→ {a.out}', flush=True)
    return 0 if res['disjoint'] else 1


def selftest():
    import contextlib
    import io
    import tempfile
    from argparse import Namespace
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / 'landmarks'
        cache.mkdir()
        shared = np.arange(9 * 59 * 3, dtype=np.float32).reshape(9, 59, 3)
        other = (np.zeros((9, 59, 3), np.float32) + 7) * 3 + 1

        def inv(name, ids_):
            p = Path(tmp) / name
            p.write_text(json.dumps({'phrases': [{'id': i, 'phrase': 'тест', 'duration': 1.0}
                                                 for i in ids_]}), encoding='utf-8')
            return str(p)
        for i, f in (('t1', shared), ('t2', other), ('e1', shared), ('e2', other * 2)):
            np.savez(cache / f'{i}.npz', frames=f)
        args = Namespace(a=inv('a.json', ['t1', 't2']), b=inv('b.json', ['e1', 'e2']),
                         cache=str(cache), window=4, out=None)
        quiet = contextlib.redirect_stdout(io.StringIO())
        with quiet:
            assert run(args) == 1, 'один и тот же жест под разными id — утечка'
        np.savez(cache / 'e1.npz', frames=other + 100)
        with quiet:
            assert run(args) == 0, 'разные записи — пересечения нет'
        with quiet:
            assert run(Namespace(**{**vars(args), 'b': inv('c.json', ['t1'])})) == 1, \
                'общий id должен ловиться до сравнения кадров'
    print('selftest ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='', help='инвентарь обучения')
    ap.add_argument('--b', default='', help='оценочный инвентарь')
    ap.add_argument('--cache', default='D:/hfdata/sp10_rsl/landmarks')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--out', default=None)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not (a.a and a.b):
            ap.error('--a и --b обязательны (или --selftest)')
        sys.exit(run(a))
