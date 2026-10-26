"""Словарь РЖЯ с YouTube (канал ARKMEDIA-SPB) → размеченные жесты в точках Holistic.

В словарных роликах слово подписано белым в левом верхнем углу ровно на время показа жеста:
отрезок = кадры с одной и той же подписью, метку один раз на отрезок читает tesseract.
Видео лежит в D:\\hfdata\\ark_dict (вне репо); в датасет идут только точки, кадры — никуда.
"""
import argparse
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live_signs import NPTS, holistic_array  # noqa: E402

CHANNEL = 'https://www.youtube.com/@ARKMEDIA-SPB/videos'
WANT = re.compile(r'словар|dictionar|vocabul|sovar', re.I)
SKIP = re.compile(r'библ|biblic|relig|религ', re.I)
WORK = Path(r'D:\hfdata\ark_dict')
STEP = 2  # каждый второй кадр: 12.5 к/с, модель всё равно пересэмплирует жест до фиксированной длины


def caption_crop(frame):
    """Левый верхний угол, минимум по каналам: белый текст светлый, синий фон тёмный."""
    h, w = frame.shape[:2]
    return frame[:int(h * .35), :w // 2].min(2)


def caption_mask(crop):
    return cv2.resize((crop > 190).astype(np.float32), (64, 32), interpolation=cv2.INTER_AREA) > .1


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def spans(masks, min_px=12, same=.7, min_len=4):
    """Индексы [a, b) подряд идущих кадров с одной и той же подписью.

    Верх и низ подписи сравниваются отдельно: «Вариант 2» под словом — мелкая строка, но это
    уже другой жест того же слова."""
    out, a = [], None
    for i, m in enumerate(masks + [None]):
        on = m is not None and m.sum() >= min_px
        if a is not None and (not on or min(iou(m[:16], masks[a][:16]), iou(m[16:], masks[a][16:])) < same):
            if i - a >= min_len:
                out.append((a, i))
            a = None
        if on and a is None:
            a = i
    return out


def read_caption(crop):
    import pytesseract
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    txt = pytesseract.image_to_string(255 - crop, lang='rus', config='--psm 6')
    lines = [re.sub(r'[^А-Яа-яЁё0-9\- ]', '', s).strip() for s in txt.splitlines()]
    lines = [s for s in lines if len(s) > 1]
    variant = next((s for s in lines if s.lower().startswith('вариант')), None)
    words = [s for s in lines if s is not variant]
    return (words[0] if words else ''), variant


def split_variants(a, b, label, every=12):
    """Делит отрезок [a, b) там, где меняется прочитанная подпись (номер варианта — одна цифра,
    по пикселям её смена тонет в шуме сжатия). OCR раз в ~1 с, смену уточняет двоичный поиск."""
    grid = list(range(a, b, every)) + [b - 1]
    labs = [label(i) for i in grid]
    cuts = []
    for i, j, x, y in zip(grid, grid[1:], labs, labs[1:]):
        if x != y:
            while j - i > 1:
                m = (i + j) // 2
                i, j = (m, j) if label(m) == x else (i, m)
            cuts.append(j)
    edges = [a] + cuts + [b]
    out = []
    for s, e in zip(edges, edges[1:]):
        lab = label(s + (e - s) // 2)
        if out and out[-1][2] == lab:  # шум OCR внутри одного варианта
            out[-1] = (out[-1][0], e, lab)
        else:
            out.append((s, e, lab))
    return out


def extract(path):
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    crops, pts = [], []
    det = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % STEP == 0:
            crops.append(caption_crop(fr))
            pts.append(holistic_array(det.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
        i += 1
    signs = []
    memo = {}

    def label(i):
        if i not in memo:
            memo[i] = read_caption(crops[i])
        return memo[i]

    for a0, b0 in spans([caption_mask(c) for c in crops]):
        for a, b, (word, variant) in split_variants(a0, b0, label):
            seg = [p for p in pts[a:b] if p is not None]
            if word and len(seg) >= 4:
                signs.append((word, variant or '', a * STEP, b * STEP, np.stack(seg)))
    return signs


def process(vid):
    out = WORK / f'{vid}.npz'
    if out.exists():
        return vid, 'cached'
    # B31: сбой одного ролика (403, tesseract OOM) не должен ронять весь прогон
    try:
        src = next(WORK.glob(f'{vid}.*'), None)
        if src is None or src.suffix == '.part':
            subprocess.run(['yt-dlp', '-q', '--no-warnings', '-f', 'bv*[height<=480]+ba/b[height<=480]',
                            '-o', str(WORK / '%(id)s.%(ext)s'), f'https://www.youtube.com/watch?v={vid}'], check=True)
            src = next(p for p in WORK.glob(f'{vid}.*') if p.suffix != '.npz')
        signs = extract(src)
        np.savez_compressed(out, frames=np.concatenate([s[4] for s in signs]) if signs else np.zeros((0, NPTS, 3)),
                            lens=np.array([len(s[4]) for s in signs]), words=np.array([s[0] for s in signs]),
                            variants=np.array([s[1] for s in signs]),
                            span=np.array([(s[2], s[3]) for s in signs]).reshape(-1, 2))
    except Exception as e:
        return vid, f'FAIL {type(e).__name__}: {str(e)[:100]}'
    return vid, f'{len(signs)} жестов'


def channel_ids():
    lines = subprocess.run(['yt-dlp', '--flat-playlist', '--print', '%(id)s|%(title)s', CHANNEL],
                           capture_output=True, text=True, encoding='utf-8', check=True).stdout.splitlines()
    return [(v, t) for v, t in (s.split('|', 1) for s in lines) if WANT.search(t) and not SKIP.search(t)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='*', help='только эти ролики (по умолчанию весь словарь канала)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=str(ROOT / 'datasets' / 'youtube_dict' / 'ark_dict.npz'))
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    vids = [(v, v) for v in a.ids] if a.ids else channel_ids()
    print(f'роликов: {len(vids)}', flush=True)
    titles = dict(vids)
    with ProcessPoolExecutor(a.workers) as ex:
        for vid, msg in ex.map(process, [v for v, _ in vids]):
            print(vid, titles[vid][:60], msg, flush=True)

    frames, offsets, words, variants, video = [], [0], [], [], []
    missing = []
    for vid, _ in vids:
        p = WORK / f'{vid}.npz'
        if not p.exists():
            missing.append(vid)
            continue
        d = np.load(p)
        frames.append(d['frames'])
        for n in d['lens']:
            offsets.append(offsets[-1] + int(n))
        words += d['words'].tolist()
        variants += d['variants'].tolist()
        video += [vid] * len(d['lens'])
    if missing:
        print(f'нет npz у {len(missing)} роликов (пропущены): {" ".join(missing)}')
    if not frames:
        print('нет данных для сборки')
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, frames=np.concatenate(frames).astype(np.float32), offsets=np.array(offsets),
                        words=np.array(words), variants=np.array(variants), video=np.array(video))
    print(f'жестов {len(words)}, разных слов {len({w.lower() for w in words})} → {a.out}')


if __name__ == '__main__':
    m = [np.zeros((32, 64), bool)] * 3 + [np.ones((32, 64), bool)] * 5 + [np.eye(32, 64, dtype=bool)] * 6
    assert spans(m) == [(3, 8), (8, 14)]
    lab = [None] * 5 + ['x'] * 20 + ['y'] * 7
    assert split_variants(5, 32, lambda i: lab[i]) == [(5, 25, 'x'), (25, 32, 'y')]
    sys.exit(main())
