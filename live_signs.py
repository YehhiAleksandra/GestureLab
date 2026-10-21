"""Нарезка живого потока ландмарок на отдельные жесты (только numpy — тестируется в CI без torch).

Клипы Slovo состоят только из кадров, где видна рука, поэтому жест = непрерывный отрезок
кадров с руками, а пауза между жестами = руки опущены из кадра на `gap` кадров подряд.
"""
import numpy as np


def hands_array(multi_hand_landmarks):
    """MediaPipe → [2, 21, 3], руки по x запястья слева направо, NaN — руки нет; None — рук нет."""
    if not multi_hand_landmarks:
        return None
    hands = sorted(([(p.x, p.y, p.z) for p in h.landmark] for h in multi_hand_landmarks[:2]),
                   key=lambda h: h[0][0])
    out = np.full((2, 21, 3), np.nan, np.float32)
    out[:len(hands)] = hands
    return out


class SignSegmenter:
    def __init__(self, gap=6, min_len=8):
        self.gap, self.min_len = gap, min_len
        self.buf, self.miss = [], 0

    def push(self, frame):
        """Кадр [2, 21, 3] или None. Возвращает законченный жест [n, 2, 21, 3] или None."""
        if frame is not None:
            self.buf.append(frame)
            self.miss = 0
            return None
        if not self.buf:
            return None
        self.miss += 1
        return self.flush() if self.miss >= self.gap else None

    def flush(self):
        seg, self.buf, self.miss = self.buf, [], 0
        return np.stack(seg) if len(seg) >= self.min_len else None
