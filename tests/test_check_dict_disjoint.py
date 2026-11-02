import numpy as np
from argparse import Namespace

from tools.check_dict_disjoint import dict_clips, evalfingerprints, run


def _args(tmp, clips):
    return Namespace(dict=str(tmp), clips=str(clips), window=4, out=None)


def test_leak_is_detected_when_dict_clip_lives_inside_eval_video(tmp_path):
    shared = np.arange(9 * 59 * 3, dtype=np.float32).reshape(9, 59, 3)
    d = tmp_path / 'dict.npz'
    np.savez(d, frames=shared, lens=np.array([9]), words=np.array(['мама']))
    clips = tmp_path / 'eval'
    clips.mkdir()
    np.savez(clips / 'r1.npz',
             frames=np.concatenate([np.zeros((5, 59, 3), np.float32), shared]))
    assert run(_args(d, clips)) == 1


def test_different_source_of_same_gestures_is_disjoint(tmp_path):
    a = np.arange(9 * 59 * 3, dtype=np.float32).reshape(9, 59, 3)
    b = a[::-1] + 0.5  # те же жесты, другая запись
    d = tmp_path / 'dict.npz'
    np.savez(d, frames=a, lens=np.array([9]), words=np.array(['мама']))
    clips = tmp_path / 'eval'
    clips.mkdir()
    np.savez(clips / 'r1.npz', frames=b)
    assert run(_args(d, clips)) == 0


def test_short_clips_are_skipped(tmp_path):
    d = tmp_path / 'dict.npz'
    np.savez(d, frames=np.zeros((6, 59, 3), np.float32), lens=np.array([3, 3]),
             words=np.array(['ма', 'па']))
    assert dict_clips(d) == []


def test_eval_fingerprints_cover_every_window(tmp_path):
    clips = tmp_path / 'eval'
    clips.mkdir()
    f = np.arange(6 * 59 * 3, dtype=np.float32).reshape(6, 59, 3)
    np.savez(clips / 'r1.npz', frames=f)
    np.savez(clips / 'r2.npz', frames=f * 2)
    fp, n = evalfingerprints(clips, window=4)
    assert n == 2 and len(fp) == 2 * (6 - 4 + 1)
