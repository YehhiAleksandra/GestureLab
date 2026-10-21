from types import SimpleNamespace

import numpy as np
import pytest

from live_signs import NPTS, SignSegmenter, hands_array, holistic_array, holistic_frame


def _hand(x):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=0.5, z=0.0)] * 21)


def test_hands_sorted_by_wrist_x_and_nan_for_missing():
    a = hands_array([_hand(0.8), _hand(0.2)])
    assert a.shape == (2, 21, 3) and a[0, 0, 0] == np.float32(0.2) and a[1, 0, 0] == np.float32(0.8)
    one = hands_array([_hand(0.5)])
    assert np.isnan(one[1]).all() and not np.isnan(one[0]).any()
    assert hands_array(None) is None


def test_segmenter_splits_on_lowered_hands_and_drops_flicker():
    f = np.zeros((2, 21, 3), np.float32)
    seg = SignSegmenter(gap=3, min_len=4)
    stream = [f] * 10 + [None] * 2 + [f] * 5 + [None] * 3 + [f] * 2 + [None] * 3
    out = [s for s in map(seg.push, stream) if s is not None]
    assert [len(s) for s in out] == [15]  # пауза в 2 кадра не режет жест, 2 кадра — дрожь
    assert seg.buf == []


def _pts(n, x):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=0.5, z=0.0)] * n)


def test_holistic_frame_layout():
    res = SimpleNamespace(left_hand_landmarks=_hand(0.7), right_hand_landmarks=None,
                          pose_landmarks=_pts(33, 0.4), face_landmarks=None)
    f = holistic_array(res)
    assert f.shape == (NPTS, 3)
    assert f[0, 0] == np.float32(0.7) and np.isnan(f[21:42]).all()
    assert (f[42:49, 0] == np.float32(0.4)).all() and np.isnan(f[49:]).all()
    assert holistic_array(SimpleNamespace(left_hand_landmarks=None, right_hand_landmarks=None)) is None
    two = holistic_frame(np.full((21, 3), 0.9), np.full((21, 3), 0.1), None, None)
    assert two[0, 0] == np.float32(0.1) and two[21, 0] == np.float32(0.9)


def test_mirror_swaps_left_and_right_points():
    torch = pytest.importorskip('torch')  # noqa: F841 — slovo_model тянет torch, в CI его нет
    from slovo_model import MIRROR, holistic_features
    assert sorted(MIRROR) == list(range(NPTS)) and (MIRROR[MIRROR] == np.arange(NPTS)).all()
    assert MIRROR[43] == 44 and MIRROR[53] == 54 and MIRROR[0] == 21
    seq = np.random.default_rng(0).random((5, NPTS, 3)).astype(np.float32)
    assert np.isfinite(holistic_features(seq)).all()
