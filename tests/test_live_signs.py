from types import SimpleNamespace

import numpy as np

from live_signs import SignSegmenter, hands_array


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
