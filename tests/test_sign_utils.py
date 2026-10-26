import numpy as np

from live_signs import NPTS
from sign_utils import (FEAT, FEAT_MOTION, features, motion_signal, split_utterances,
                        stream_features)

L_SHOULDER, R_SHOULDER = 43, 44  # POSE_IDX[1], POSE_IDX[2]


def _frame(left=None, right=None, shoulder=(0.4, 0.6), eye=0.05):
    """Holistic-кадр: кисти в (x, 0.5), плечи на y=0.6, глаза на y=0.3."""
    f = np.full((NPTS, 3), np.nan, np.float32)
    for k, wrist in enumerate((left, right)):
        if wrist is not None:
            f[21 * k:21 * k + 21, 0] = wrist
            f[21 * k:21 * k + 21, 1] = 0.5
            f[21 * k:21 * k + 21, 2] = 0.0
    f[L_SHOULDER] = (shoulder[0], 0.6, 0.0)
    f[R_SHOULDER] = (shoulder[1], 0.6, 0.0)
    f[55] = (0.5 - eye, 0.3, 0.0)  # углы глаз
    f[56] = (0.5 + eye, 0.3, 0.0)
    f[51] = (0.48, 0.4, 0.0)  # рот
    f[52] = (0.52, 0.4, 0.0)
    f[53] = (0.45, 0.42, 0.0)
    f[54] = (0.55, 0.42, 0.0)
    return f


def _stream(t, left=lambda i: 0.35, right=lambda i: 0.65):
    return np.stack([_frame(left(i), right(i)) for i in range(t)])


def test_stream_features_shape_and_finite():
    out = stream_features(_stream(12))
    assert out.shape == (12, FEAT) and np.isfinite(out).all()


def test_features_motion_appends_frame_delta():
    seq = _stream(6, left=lambda i: 0.35 + 0.01 * i)
    base = features(seq)
    full = features(seq, FEAT_MOTION)
    assert base.shape == (6, FEAT) and np.array_equal(base, stream_features(seq))
    assert full.shape == (6, FEAT_MOTION)
    assert np.allclose(full[0, FEAT:], 0)  # у первого кадра предыдущего нет
    assert np.allclose(full[1:, FEAT:], base[1:] - base[:-1])
    try:
        features(seq, FEAT + 1)
    except ValueError:
        pass
    else:
        raise AssertionError('неизвестная ширина должна падать')


def test_stream_features_body_coords_are_shift_invariant():
    a = _stream(5)
    b = a.copy()
    b[..., :2] += np.array([0.1, -0.05], np.float32)  # сдвиг всей сцены
    assert np.allclose(stream_features(a), stream_features(b), atol=1e-5)


def test_stream_features_without_pose_or_face_falls_back():
    s = _stream(4)
    s[:, L_SHOULDER:] = np.nan
    out = stream_features(s)
    assert out.shape == (4, FEAT) and np.isfinite(out).all()


def test_stream_features_no_hands_marks_absence():
    s = np.stack([_frame(None, None)] * 3)
    out = stream_features(s)
    assert np.isfinite(out).all()
    assert (out[:, -2:] == 0).all()  # оба флага присутствия рук


def test_motion_signal_zero_for_still_hands():
    assert np.allclose(motion_signal(_stream(6)), 0, atol=1e-6)


def test_motion_signal_grows_with_wrist_speed():
    still = motion_signal(_stream(6))
    moving = motion_signal(_stream(6, left=lambda i: 0.3 + 0.02 * i))
    assert moving.max() > 0 and moving.mean() > still.mean()


def test_split_utterances_cuts_on_hand_absence():
    fps = 25
    frames = np.concatenate([_stream(21), np.stack([_frame()] * 25), _stream(35)])
    spans = split_utterances(frames, fps, pause=0.7)
    assert spans == [(0, 21), (46, 81)]


def test_split_utterances_drops_short_span():
    frames = np.concatenate([_stream(3), np.stack([_frame()] * 25), _stream(30)])
    assert split_utterances(frames, 25, pause=0.7) == [(28, 58)]


def test_split_utterances_cuts_long_span_by_max_len():
    fps = 10
    frames = _stream(60, left=lambda i: 0.3 + 0.001 * i)
    spans = split_utterances(frames, fps, pause=0.7, max_len=2.0)
    assert all(b - a <= 20 for a, b in spans)
    assert spans[0][0] == 0 and spans[-1][1] == 60
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
