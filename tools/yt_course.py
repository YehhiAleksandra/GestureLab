"""Курс РЖЯ с YouTube (плейлист PLA357506657984F1A) → размеченные жесты в точках Holistic.

Преподаватель сидит за столом: называет слово голосом, поднимает руки, показывает жест и
кладёт руки на стол. Руки видны всё время, поэтому жест = отрезок, где запястье поднято над
уровнем покоя; метка — однословная реплика из расшифровки (`reports/ref/*.json`, whisper),
начавшаяся незадолго до отрезка или внутри него. Разные слова рядом — отрезок выброшен.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live_signs import NPTS, holistic_array  # noqa: E402

VIDEOS = ROOT / 'datasets' / 'youtube' / 'Курс русского жестового языка'
REFS = ROOT / 'reports' / 'ref'
WORK = Path(r'D:\hfdata\course')
STEP = 2
LEAD = 2.0  # слово произносится до жеста не раньше чем за столько секунд


def raised(height, margin=.35, min_len=6, gap=3):
    """height — (y запястья − y плеч)/ширина плеч, NaN без позы. Отрезки [a, b), где запястье
    выше медианного уровня покоя на margin ширин плеч; разрывы короче gap кадров заклеиваются."""
    rest = np.nanmedian(height)
    on = np.nan_to_num(height, nan=rest) < rest - margin
    out, a, miss = [], None, 0
    for i, v in enumerate(list(on) + [False] * (gap + 1)):
        if v:
            a = i if a is None else a
            miss = 0
        elif a is not None:
            miss += 1
            if miss > gap:
                if i - miss + 1 - a >= min_len:
                    out.append((a, i - miss + 1))
                a, miss = None, 0
    return out


def one_word(text):
    w = re.sub(r'[^а-яё\- ]', '', text.lower()).split()
    return w[0] if len(w) == 1 else None


def label_runs(runs, segs, fps):
    """runs — отрезки в кадрах (после STEP), segs — [{'start', 'text'}] → [(a, b, слово)]."""
    words = [(s['start'], one_word(s['text'])) for s in segs]
    out = []
    for a, b in runs:
        t0, t1 = a / fps, b / fps
        near = {w for t, w in words if w and t0 - LEAD <= t <= t1}
        if len(near) == 1:
            out.append((a, b, near.pop()))
    return out


def extract(path):
    import mediapipe as mp
    det = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) / STEP
    pts, height, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % STEP == 0:
            res = det.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            pts.append(holistic_array(res))
            h = np.nan
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                width = abs(lm[11].x - lm[12].x) or np.nan
                h = (min(lm[15].y, lm[16].y) - (lm[11].y + lm[12].y) / 2) / width
            height.append(h)
        i += 1
    return pts, np.array(height, np.float32), fps


def process(path):
    out = WORK / f'{path.stem[:3]}.npz'
    if out.exists():
        return path.stem, 'cached'
    segs = json.loads((REFS / f'{path.stem}.json').read_text(encoding='utf-8'))
    pts, height, fps = extract(path)
    signs = []
    for a, b, w in label_runs(raised(height), segs, fps):
        seg = [p for p in pts[a:b] if p is not None]
        if len(seg) >= 4:
            signs.append((w, a / fps, b / fps, np.stack(seg)))
    np.savez_compressed(out, frames=np.concatenate([s[3] for s in signs]) if signs else np.zeros((0, NPTS, 3)),
                        lens=np.array([len(s[3]) for s in signs]), words=np.array([s[0] for s in signs]),
                        span=np.array([(s[1], s[2]) for s in signs]).reshape(-1, 2))
    return path.stem, f'{len(signs)} жестов из {len(raised(height))} подъёмов рук'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lessons', nargs='*', type=int, help='номера уроков (по умолчанию все)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=str(ROOT / 'datasets' / 'youtube_dict' / 'course.npz'))
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    paths = sorted(VIDEOS.glob('*.mp4'))
    if a.lessons:
        paths = [p for p in paths if int(p.stem[:3]) in a.lessons]
    with ProcessPoolExecutor(a.workers) as ex:
        for stem, msg in ex.map(process, paths):
            print(stem, msg, flush=True)

    frames, offsets, words, video = [], [0], [], []
    for p in paths:
        d = np.load(WORK / f'{p.stem[:3]}.npz')
        frames.append(d['frames'])
        for n in d['lens']:
            offsets.append(offsets[-1] + int(n))
        words += d['words'].tolist()
        video += [p.stem[:3]] * len(d['lens'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, frames=np.concatenate(frames).astype(np.float32), offsets=np.array(offsets),
                        words=np.array(words), video=np.array(video))
    print(f'жестов {len(words)}, разных слов {len(set(words))} → {a.out}')


if __name__ == '__main__':
    h = np.array([0, 0, 0, -1, -1, -1, np.nan, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0], np.float32) + 2
    assert raised(h, margin=.5, min_len=3, gap=1) == [(3, 10)]
    assert label_runs([(10, 20)], [{'start': 0.5, 'text': 'Хлеб.'}, {'start': 1.6, 'text': 'Хлеб.'},
                                   {'start': 3.0, 'text': 'Булка.'}], fps=10) == [(10, 20, 'хлеб')]
    sys.exit(main())
