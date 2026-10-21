"""Нарезка живого потока ландмарок на отдельные жесты (только numpy — тестируется в CI без torch).

Клипы Slovo состоят только из кадров, где видна рука, поэтому жест = непрерывный отрезок
кадров с руками, а пауза между жестами = руки опущены из кадра на `gap` кадров подряд.
"""
import numpy as np

# Holistic-кадр [NPTS, 3]: 2×21 точка рук (по x запястья) + POSE_IDX + FACE_IDX.
# Индексы лица < 468 — общие для FaceMesh legacy (468) и MediaPipe Tasks (478).
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # нос, плечи, локти, запястья
FACE_IDX = [1, 152, 13, 14, 61, 291, 33, 263, 105, 334]  # нос, подбородок, губы, углы рта, глаза, брови
NPTS = 42 + len(POSE_IDX) + len(FACE_IDX)


def holistic_frame(hand_a, hand_b, pose, face):
    """Части в долях кадра ([21,3], [len(POSE_IDX),3], [len(FACE_IDX),3]) или None → [NPTS, 3]."""
    out = np.full((NPTS, 3), np.nan, np.float32)
    hands = sorted((h for h in (hand_a, hand_b) if h is not None), key=lambda h: h[0][0])
    for i, h in enumerate(hands):
        out[21 * i:21 * i + 21] = h
    if pose is not None:
        out[42:49] = pose
    if face is not None:
        out[49:] = face
    return out


def holistic_array(res):
    """Результат mp.solutions.holistic → [NPTS, 3]; None, если ни одной руки нет."""
    if res.left_hand_landmarks is None and res.right_hand_landmarks is None:
        return None

    def pts(lm, idx=slice(None)):
        return None if lm is None else np.array([(p.x, p.y, p.z) for p in lm.landmark], np.float32)[idx]
    return holistic_frame(pts(res.left_hand_landmarks), pts(res.right_hand_landmarks),
                          pts(res.pose_landmarks, POSE_IDX), pts(res.face_landmarks, FACE_IDX))


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
        """Кадр ([2, 21, 3] или holistic [NPTS, 3]) или None. Возвращает законченный жест [n, ...] или None."""
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
