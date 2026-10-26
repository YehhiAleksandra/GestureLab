"""Скелет MediaPipe Holistic поверх видео: ровно те точки, которые читает CTC-модель.

Рисуются 2×21 точки кистей, POSE_IDX (нос, плечи, локти, запястья) и FACE_IDX
(10 точек рта/глаз/бровей) — остальные 458 точек лица модель не видит и не рисуются.
Пишет немое видео; звук и субтитры накладываются ffmpeg.

  python tools/skeleton_overlay.py --video D:\\...\\vazhno.mp4 --out D:\\...\\skel_raw.mp4
  python tools/skeleton_overlay.py --selftest
"""
import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gl_capture import open_video  # noqa: E402
from live_signs import FACE_IDX, POSE_IDX  # noqa: E402

HAND_LINKS = list(mp.solutions.hands.HAND_CONNECTIONS)
# позиции внутри POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
POSE_LINKS = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]
# позиции внутри FACE_IDX = [1, 152, 13, 14, 61, 291, 33, 263, 105, 334]
FACE_LINKS = [(4, 2), (2, 5), (4, 3), (3, 5), (6, 7)]

LEFT = (60, 220, 60)      # BGR: зелёный
RIGHT = (40, 170, 255)    # оранжевый
POSE = (235, 235, 235)    # белый
FACE = (255, 220, 60)     # голубой


def _xy(p, w, h):
    return int(round(p[0] * w)), int(round(p[1] * h))


def _line(img, a, b, color, t=2):
    if np.isnan(a).any() or np.isnan(b).any():
        return
    h, w = img.shape[:2]
    p1, p2 = _xy(a, w, h), _xy(b, w, h)
    cv2.line(img, p1, p2, (0, 0, 0), t + 2, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, t, cv2.LINE_AA)


def _dot(img, p, color, r=3):
    if np.isnan(p).any():
        return
    h, w = img.shape[:2]
    c = _xy(p, w, h)
    cv2.circle(img, c, r + 1, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, c, r, color, -1, cv2.LINE_AA)


def draw_holistic(img, left, right, pose, face):
    """Точки в долях кадра: [21,3] руки, [7,3] поза, [10,3] лицо; None/NaN пропускаются."""
    # поза и лицо первыми: кисти должны быть поверх корпуса
    if pose is not None:
        for a, b in POSE_LINKS:
            _line(img, pose[a], pose[b], POSE, 2)
        for p in pose:
            _dot(img, p, POSE, 2)
    if face is not None:
        for a, b in FACE_LINKS:
            _line(img, face[a], face[b], FACE, 1)
        for p in face:
            _dot(img, p, FACE, 2)
    for hand, color in ((left, LEFT), (right, RIGHT)):
        if hand is None:
            continue
        for a, b in HAND_LINKS:
            _line(img, hand[a], hand[b], color, 2)
        for p in hand:
            _dot(img, p, color, 3)
    return img


def _pts(lm):
    if lm is None:
        return None
    return np.array([(p.x, p.y, p.z) for p in lm.landmark], np.float32)


def overlay(video, out):
    cap = open_video(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, first = cap.read()
    if not ok:
        sys.exit(f'в видео нет кадров: {video}')
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                              smooth_landmarks=True)
    frames = hands = poses = 0
    try:
        frame = first
        while ok:
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose = None if res.pose_landmarks is None else _pts(res.pose_landmarks)[POSE_IDX]
            face = None if res.face_landmarks is None else _pts(res.face_landmarks)[FACE_IDX]
            draw_holistic(frame, _pts(res.left_hand_landmarks), _pts(res.right_hand_landmarks),
                          pose, face)
            writer.write(frame)
            frames += 1
            hands += res.left_hand_landmarks is not None or res.right_hand_landmarks is not None
            poses += res.pose_landmarks is not None
            ok, frame = cap.read()
    finally:
        holistic.close()
        cap.release()
        writer.release()
    print(f'кадров {frames}, с кистью {hands} ({100 * hands / max(frames, 1):.0f}%), '
          f'с позой {poses} ({100 * poses / max(frames, 1):.0f}%) → {out}', flush=True)


def selftest():
    t = np.linspace(0.2, 0.8, 21, dtype=np.float32)
    hand = np.stack([t, t[::-1], np.zeros(21, np.float32)], 1)
    pose = hand[:7]
    face = hand[:10]
    img = np.zeros((120, 160, 3), np.uint8)
    draw_holistic(img, hand, None, pose, face)
    assert img.any(), 'скелет не нарисован'
    nan = np.full((21, 3), np.nan, np.float32)
    draw_holistic(np.zeros_like(img), nan, nan, nan, nan)  # NaN не должны падать
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', help='входное видео, путь C:\\... или D:\\...')
    ap.add_argument('--out', help='выходное немое видео')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.video or not a.out:
        sys.exit('нужны --video и --out')
    overlay(a.video, a.out)


if __name__ == '__main__':
    main()
