"""Гейт утечки: клипы обучающего словаря не должны попадать в оценочные ролики.

Причина: все честные замеры этапа 21 опираются на утверждение, что словарь канала
(YouTube-ролики ark_*) и фразовые клипы (Rutube) — разные записи. Если это перестанет
быть правдой, «отложенная» оценка молча превратится в самопроверку, как уже случилось с
дефолтным `--init` (B62). Проверка сравнивает не имена файлов, а байты ландмарок: источник
меток может переименовываться, а один и тот же жест из другого ролика — не утечка.

  python tools/check_dict_disjoint.py --selftest
  python tools/check_dict_disjoint.py --dict D:/hfdata/ark_all.npz \
      --clips D:/hfdata/rutube_phrases/landmarks --out reports/evidence/dict_phrase_disjoint.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _h(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def dict_clips(path):
    """npz словаря (frames + lens или offsets + words) → список кадровых массивов."""
    with np.load(path, allow_pickle=True) as d:
        frames = d['frames']
        lens = (d['lens'].astype(int) if 'lens' in d
                else np.diff(d['offsets'].astype(int)))
        words = [str(x) for x in d['words']]
    starts = np.concatenate([[0], np.cumsum(lens)])
    return [(words[k], frames[starts[k]:starts[k + 1]])
            for k in range(len(lens)) if starts[k + 1] - starts[k] >= 4]


def evalfingerprints(dirpath, window=4):
    """Оценка → ({отпечаток окна: имя файла}, число роликов, суммарно окон)."""
    fp = {}
    n = 0
    for p in sorted(Path(dirpath).glob('*.npz')):
        with np.load(p) as d:
            f = np.ascontiguousarray(d['frames'])
        n += 1
        for i in range(len(f) - window + 1):
            fp.setdefault(_h(f[i:i + window]), p.name)
    return fp, n


def run(a):
    clips = dict_clips(a.dict)
    fp, n_eval = evalfingerprints(a.clips, a.window)
    exact, window_hits, examples = 0, 0, []
    for w, seg in clips:
        if _h(seg) in fp:
            exact += 1
            examples.append({'word': w, 'kind': 'полный клип'})
        elif _h(seg[:a.window]) in fp:
            window_hits += 1
            if len(examples) < 20:
                examples.append({'word': w, 'kind': f'окно {a.window} кадров'})
    res = {'tool': 'check_dict_disjoint', 'date': time.strftime('%Y-%m-%d'),
           'dict': str(a.dict), 'dict_clips': len(clips), 'eval_clips': n_eval,
           'window': a.window, 'overlap_exact': exact, 'overlap_window': window_hits,
           'disjoint': exact == 0 and window_hits == 0, 'examples': examples[:20]}
    print(json.dumps({k: res[k] for k in
                      ('dict_clips', 'eval_clips', 'overlap_exact', 'overlap_window',
                       'disjoint')}, ensure_ascii=False), flush=True)
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
        shared = np.arange(9 * 59 * 3, dtype=np.float32).reshape(9, 59, 3)
        other = np.zeros((9, 59, 3), np.float32) + 7
        np.savez(Path(tmp) / 'dict.npz', frames=np.concatenate([shared, other]),
                 lens=np.array([9, 9]), words=np.array(['мама', 'папа']))
        clips = Path(tmp) / 'eval'
        clips.mkdir()
        args = Namespace(dict=str(Path(tmp) / 'dict.npz'), clips=str(clips), window=4, out=None)
        quiet = contextlib.redirect_stdout(io.StringIO())
        # ролик, внутри которого лежит словарный клип, — утечка должна ловиться
        np.savez(clips / 'r1.npz', frames=np.concatenate([np.zeros((3, 59, 3), np.float32),
                                                          shared]))
        with quiet:
            assert run(args) == 1
        # другой источник тех же жестов — пересечения нет
        np.savez(clips / 'r1.npz', frames=other * 3 + 1)
        with quiet:
            assert run(args) == 0
    print('selftest ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict', default='D:/hfdata/ark_all.npz')
    ap.add_argument('--clips', default='D:/hfdata/rutube_phrases/landmarks')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--out', default=None)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        sys.exit(run(args))
