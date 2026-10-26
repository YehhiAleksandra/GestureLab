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
from live_signs import holistic_array

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
            capture_output=True, timeout=300, encoding='utf-8', errors='replace', check=False,
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
        capture_output=True, timeout=300, encoding='utf-8', errors='replace', check=False,
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


def merge(out_dir, out_path, video_ids=None):
    selected = None if video_ids is None else set(video_ids)
    all_frames, all_lens, all_words, all_videos = [], [], [], []
    for p in sorted(Path(out_dir).glob('*.npz')):
        if selected is not None and p.stem not in selected:
            continue
        with np.load(str(p)) as d:
            frames = d['frames']
            all_frames.append(frames)
            all_lens.append(len(frames))
            all_words.append(str(d['word']))
            all_videos.append(p.stem)

    if not all_frames:
        print('Нет данных')
        return 0

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frames=np.concatenate(all_frames),
        lens=np.array(all_lens),
        words=np.array(all_words),
        video=np.array(all_videos),
    )
    print(f'Сохранено {out_path}: {len(all_words)} жестов, '
          f'{sum(all_lens)} кадров', flush=True)
    return len(all_words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default='datasets/youtube_dict/rutube_dict.npz')
    ap.add_argument('--limit', type=int, default=0, help='обработать только N первых (0=все)')
    ap.add_argument('--merge-only', action='store_true',
                    help='только собрать --out из уже готового кеша, без скачивания')
    a = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    out_dir = WORK / 'npz'
    out_dir.mkdir(exist_ok=True)

    if a.merge_only:
        merge(out_dir, a.out, video_ids=None)
        return

    videos = list_videos()
    print(f'Словарных видео: {len(videos)}', flush=True)
    if a.limit > 0:
        videos = videos[:a.limit]

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(extract_landmarks, vid, word, out_dir): (vid, word)
                for vid, word in videos}
        for done, f in enumerate(as_completed(futs), 1):
            _, result = f.result()
            if result not in ('cached', 'download_fail', 'too_short'):
                print(f'{done}/{len(videos)} {result}', flush=True)

    merge(out_dir, a.out, video_ids=[vid for vid, _ in videos])


if __name__ == '__main__':
    main()
