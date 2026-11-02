"""Гейт утечки между сплитами одного бенчмарка: обучение и оценка — разные записи."""
import json
from argparse import Namespace

import numpy as np

from tools.check_split_disjoint import ids, run, windows


def _inv(path, names):
    path.write_text(json.dumps({'phrases': [
        {'id': n, 'phrase': 'дом', 'duration': (None if n.endswith('_none') else 1.0)}
        for n in names]}), encoding='utf-8')
    return str(path)


def _cache(tmp_path, frames):
    cache = tmp_path / 'landmarks'
    cache.mkdir()
    for name, f in frames.items():
        np.savez(cache / f'{name}.npz', frames=f)
    return str(cache)


def _args(tmp_path, a_names, b_names, frames):
    return Namespace(a=_inv(tmp_path / 'a.json', a_names),
                     b=_inv(tmp_path / 'b.json', b_names),
                     cache=_cache(tmp_path, frames), window=4, out=None)


def test_same_recording_under_different_ids_is_a_leak(tmp_path):
    shared = np.arange(9 * 59 * 3, dtype=np.float32).reshape(9, 59, 3)
    args = _args(tmp_path, ['t1'], ['e1'], {'t1': shared, 'e1': shared})
    assert run(args) == 1, 'один и тот же клип под другим id — оценка смотрелась в тренировку'


def test_different_recordings_of_the_benchmark_are_disjoint(tmp_path):
    a = np.arange(9 * 59 * 3, dtype=np.float32).reshape(9, 59, 3)
    args = _args(tmp_path, ['t1', 't2'], ['e1'], {'t1': a, 't2': a * 3 + 1, 'e1': a + 100})
    assert run(args) == 0


def test_shared_id_is_caught_before_frames(tmp_path):
    """Клип с общим id в обоих инвентарях — утечка, даже если кадры не совпадают (другой кэш)."""
    args = _args(tmp_path, ['x'], ['x'], {'x': np.ones((6, 59, 3), np.float32)})
    r = run(args)
    assert r == 1


def test_inventory_ids_skip_clips_without_duration(tmp_path):
    assert ids(_inv(tmp_path / 'a.json', ['t1', 't2_none'])) == {'t1'}


def test_windows_skips_clips_missing_from_cache(tmp_path):
    frames = np.arange(6 * 59 * 3, dtype=np.float32).reshape(6, 59, 3)
    cache = _cache(tmp_path, {'t1': frames})
    fp, missing = windows({'t1', 't9'}, cache, window=4)
    assert missing == 1 and set(fp.values()) == {'t1'}
    assert len(fp) == 6 - 4 + 1, 'окон должно быть ровно len(frames)-window+1'
