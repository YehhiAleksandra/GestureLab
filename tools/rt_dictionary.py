"""Извлечение изолированных жестов из Rutube-канала «Основы жестового языка».

Формат: 1 видео = 1 слово (метка из заголовка «ЖЕСТОВЫЙ ЯЗЫК. СЛОВАРЬ. <Слово>»).
Скачивает yt-dlp, прогоняет MediaPipe Holistic, сохраняет holistic landmarks [T, 59, 3].

Пример:
  python tools/rt_dictionary.py --workers 3 --out datasets/youtube_dict/rutube_dict.npz
"""
import argparse
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live_signs import NPTS, holistic_array  # noqa: E402

CHANNEL = 'https://rutube.ru/channel/23691449/'
PREFIX = 'ЖЕСТОВЫЙ ЯЗЫК. СЛОВАРЬ. '
WORK = Path(tempfile.gettempdir()) / 'rt_dict'
FPS_SKIP = 2


def list_videos(cache='D:/hfdata/rutube_titles.txt'):
    if Path(cache).exists():
        lines = Path(cache).read_text(encoding='utf-8').strip().split('\n')
    else:
        r = subprocess.run(
            ['yt-dlp', '--flat-playlist', '--print', '%(id)s|%(title)s', CHANNEL],
            capture_output=True, timeout=300, encoding='utf-8', errors='replace',
        )
        lines = r.stdout.strip().split('\n')
        Path(cache).write_text(r.stdout, encoding='utf-8')
    out = []
    for line in lines:
        parts = line.strip().split('|', 1)
        if len(parts) != 2:
            continue
        vid, title = parts
        if PREFIX not in title:
            continue
        word = title.split(PREFIX, 1)[1].strip()
        word = re.sub(r'\s*\(.*\)', '', word).strip()
        word = re.sub(r'\.\w+$', '', word).strip()  # «быть.mpg» → «быть»
        if word:
            out.append((vid, word))
    return out


def download(vid):
    dst = WORK / f'{vid}.mp4'
    if dst.exists():
        return dst
    subprocess.run(
        ['yt-dlp', '-f', 'bestvideo[height<=480]/best[height<=480]/best',
         '--no-playlist', '-o', str(dst),
         f'https://rutube.ru/video/{vid}/'],
        capture_output=True, timeout=300, encoding='utf-8', errors='replace',
    )
    return dst if dst.exists() else None


def extract_landmarks(vid, word, out_dir):
    out = out_dir / f'{vid}.npz'
    if out.exists():
        return vid, 'cached'
    src = download(vid)
    if src is None:
        return vid, 'download_fail'

    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.3, min_tracking_confidence=0.3,
    )

    cap = cv2.VideoCapture(str(src))
    frames = []
    idx = 0
    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            break
        if idx % FPS_SKIP == 0:
            res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frame = holistic_array(res)
            if frame is not None:
                frames.append(frame)
        idx += 1
    cap.release()
    holistic.close()

    if len(frames) < 4:
        return vid, 'too_short'

    arr = np.stack(frames).astype(np.float32)
    np.savez_compressed(str(out), frames=arr, word=word)
    try:
        src.unlink()
    except OSError:
        pass
    return vid, word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default='datasets/youtube_dict/rutube_dict.npz')
    ap.add_argument('--limit', type=int, default=0, help='обработать только N первых (0=все)')
    a = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    out_dir = WORK / 'npz'
    out_dir.mkdir(exist_ok=True)

    videos = list_videos()
    print(f'Словарных видео: {len(videos)}', flush=True)
    if a.limit > 0:
        videos = videos[:a.limit]

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(extract_landmarks, vid, word, out_dir): (vid, word)
                for vid, word in videos}
        done = 0
        for f in as_completed(futs):
            done += 1
            vid, result = f.result()
            if result not in ('cached', 'download_fail', 'too_short'):
                print(f'{done}/{len(videos)} {result}', flush=True)

    all_frames, all_lens, all_words = [], [], []
    for vid, word in videos:
        p = out_dir / f'{vid}.npz'
        if not p.exists():
            continue
        d = np.load(str(p))
        all_frames.append(d['frames'])
        all_lens.append(len(d['frames']))
        all_words.append(word)

    if not all_frames:
        print('Нет данных')
        return

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        frames=np.concatenate(all_frames),
        lens=np.array(all_lens),
        words=np.array(all_words),
    )
    print(f'Сохранено {a.out}: {len(all_words)} жестов, '
          f'{sum(all_lens)} кадров', flush=True)


if __name__ == '__main__':
    main()
