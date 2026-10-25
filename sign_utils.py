"""Покадровые признаки и нарезка непрерывного потока РЖЯ (только numpy — тестируется в CI).

Модель и CTC-декодирование живут в sign_stream.py: там нужен torch, а эти функции —
граница потока и признаки — не должны тянуть его в CI.
"""
import warnings

import numpy as np

from live_signs import NPTS

FEAT = 2 * 21 * 3 + NPTS * 2 + 2 + 2  # 248


def _nearest_valid(ok, values):
    """Заменить невалидные значения ближайшими валидными по времени."""
    idx = np.where(ok)[0]
    if not len(idx):
        return None
    pos = np.clip(np.searchsorted(idx, np.arange(len(ok))), 0, len(idx) - 1)
    left = idx[np.clip(pos - 1, 0, len(idx) - 1)]
    right = idx[pos]
    src = np.where(np.abs(np.arange(len(ok)) - left) <= np.abs(right - np.arange(len(ok))),
                   left, right)
    return values[src]


def stream_features(seq):
    """[T, NPTS, 3] → [T, FEAT]: форма кистей + все точки в системе плеч кадра + рот."""
    seq = np.asarray(seq, np.float32)
    t = len(seq)
    hands = seq[:, :42].reshape(t, 2, 21, 3)
    present = ~np.isnan(hands[:, :, 0, 0])
    size = np.linalg.norm(hands[:, :, 9, :2] - hands[:, :, 0, :2], axis=-1)[..., None, None]
    local = (hands - hands[:, :, :1]) / np.maximum(size, 1e-3)

    sh = seq[:, 43:45, :2]
    with warnings.catch_warnings():  # кадры без позы дают пустой nanmean
        warnings.simplefilter('ignore', RuntimeWarning)
        center = np.nanmean(sh, 1)
    width = np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1)
    ok = np.isfinite(center).all(1) & (width > 1e-3)
    center_f = _nearest_valid(ok, center)
    width_f = _nearest_valid(ok, width)
    if center_f is None:
        center_f = np.tile(np.array([0.5, 0.6], np.float32), (t, 1))
        width_f = np.full(t, 0.25, np.float32)
    body = (seq[..., :2] - center_f[:, None, :]) / np.maximum(width_f, 1e-3)[:, None, None]

    eyes = np.linalg.norm(seq[:, 55, :2] - seq[:, 56, :2], axis=-1)
    mouth = np.stack([np.linalg.norm(seq[:, 51, :2] - seq[:, 52, :2], axis=-1),
                      np.linalg.norm(seq[:, 53, :2] - seq[:, 54, :2], axis=-1)], 1)
    mouth = mouth / np.maximum(eyes, 1e-3)[:, None]
    out = np.concatenate([local.reshape(t, -1), body.reshape(t, -1), mouth,
                          present.astype(np.float32)], 1)
    return np.nan_to_num(out, nan=0.0)


def motion_signal(frames):
    """Скорость запястий в долях ширины плеч; 0, когда рук нет или плеч нет."""
    frames = np.asarray(frames, np.float32)
    t = len(frames)
    wrists = frames[:, [0, 21], :2]  # левое/правое запястье
    w = np.zeros(t, np.float32)
    if t > 1:
        sh = frames[:, 43:45, :2]
        width = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
        width = width if np.isfinite(width) and width > 1e-3 else 0.25
        d = np.linalg.norm(np.diff(wrists, axis=0), axis=-1)
        w[1:] = np.nan_to_num(d, nan=0.0).mean(1) / width
    return w


def split_utterances(frames, fps, pause=0.7, min_len=0.6, max_len=15.0):
    """Разбить поток на реплики по паузам без рук; длинные реплики режутся по max_len.

    Возвращает [(start, end)] — границы кадров, end не включается.
    """
    frames = np.asarray(frames)
    t = len(frames)
    if not t:
        return []
    hands = frames[:, :42].reshape(t, 2, 21, 3)
    present = ~np.isnan(hands[:, :, 0, 0]).all(1)  # хотя бы одна рука
    gap = max(1, round(pause * fps))
    spans, start, miss = [], None, 0
    for i in range(t):
        if present[i]:
            if start is None:
                start = i
            miss = 0
        elif start is not None:
            miss += 1
            if miss >= gap:
                spans.append((start, i - miss + 1))
                start, miss = None, 0
    if start is not None:
        spans.append((start, t))

    out = []
    for a, b in spans:
        if b - a < min_len * fps:
            continue
        limit = round(max_len * fps)
        speed = motion_signal(frames[a:b])
        while b - a > limit:
            cut = a + limit
            # сдвигаем разрез назад к самому спокойному кадру, но не за предел max_len
            lo, hi = max(a + int(0.5 * fps), cut - int(2 * fps)), min(b, cut)
            if hi - lo > 1:
                cut = lo + int(np.argmin(speed[lo - a:hi - a]))
            out.append((a, cut))
            a = cut
        out.append((a, b))
    return out
