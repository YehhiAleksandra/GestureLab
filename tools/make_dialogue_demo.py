"""Собрать диалоговое демо из тестовых клипов Slovo: глоссы → id → видео.

Клипы берутся из `datasets/slovo/test` по id из `slovo_holistic.npz` (те же id, что
в разметке), поэтому эталон известен заранее. Видео реальных людей в репозиторий не
кладётся: mp4 собирается вне репо, в репо — только манифест с id и границами.

Реплики разделяются `|`; внутри реплики пауза короткая (слитная речь), между
репликами — длинная, чтобы сработала нарезка по отсутствию рук.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gl_capture import open_video  # noqa: E402


def plan_timeline(lengths, gaps, tail):
    """Длины клипов и паузы между ними → (начало каждого клипа, всего кадров)."""
    starts, pos = [], 0
    for n, g in zip(lengths, list(gaps) + [0]):
        starts.append(pos)
        pos += n + g
    return starts, pos + tail


def fit(frame, w, h):
    """Вписать кадр в w×h с сохранением пропорций, поля чёрные."""
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = round(fw * s), round(fh * s)
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y:y + nh, x:x + nw] = small
    return out


def pick_clips(npz_path, turns):
    """Для каждой глоссы — первый тестовый клип; возвращает [(глосса, id)]."""
    z = np.load(npz_path, allow_pickle=False)
    texts, ids, train = z['texts'], z['ids'], z['train']
    chosen, used = [], set()
    for turn in turns:
        for gloss in turn:
            hit = next((i for i in np.where(~train)[0]
                        if texts[i] == gloss and ids[i] not in used), None)
            if hit is None:
                raise SystemExit(f'нет тестового клипа для «{gloss}»')
            used.add(ids[hit])
            chosen.append((gloss, str(ids[hit])))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glosses', required=True, help='реплики через |, глоссы через запятую')
    ap.add_argument('--data', default=str(ROOT / 'datasets/slovo/slovo_holistic.npz'))
    ap.add_argument('--videos', default=str(ROOT / 'datasets/slovo/test'))
    ap.add_argument('--out', default='D:/hfdata/demo_dialog.mp4')
    ap.add_argument('--manifest', default=str(ROOT / 'reports/evidence/demo_dialog_manifest.json'))
    ap.add_argument('--gap', type=float, default=0.17, help='пауза внутри реплики, с')
    ap.add_argument('--turn-gap', type=float, default=0.8, help='пауза между репликами, с')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--size', default='640x640')
    a = ap.parse_args()

    w, h = (int(v) for v in a.size.lower().split('x'))
    turns = [[g.strip() for g in t.split(',') if g.strip()] for t in a.glosses.split('|')]
    clips = pick_clips(a.data, turns)

    # паузы между клипами: короткая внутри реплики, длинная на границе реплик
    gaps = []
    for ti, turn in enumerate(turns):
        for j in range(len(turn)):
            if ti == len(turns) - 1 and j == len(turn) - 1:
                continue
            gaps.append(a.turn_gap if j == len(turn) - 1 else a.gap)

    writer, timeline, lengths = None, [], []
    for gloss, vid in clips:
        cap = open_video(str(Path(a.videos) / f'{vid}.mp4'))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fit(fr, w, h))
        cap.release()
        if not frames:
            raise SystemExit(f'пустое видео {vid}')
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (w, h))
        lengths.append(len(frames))
        timeline.append({'gloss': gloss, 'id': vid, 'frames': len(frames)})

    starts, total = plan_timeline(lengths, [round(g * a.fps) for g in gaps],
                                  round(1.0 * a.fps))
    for (_gloss, vid), n, g in zip(clips, lengths, gaps + [0]):
        cap = open_video(str(Path(a.videos) / f'{vid}.mp4'))
        shown = 0
        while shown < n:
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(fit(fr, w, h))
            shown += 1
        cap.release()
        for _ in range(round(g * a.fps)):
            writer.write(np.zeros((h, w, 3), np.uint8))
    for _ in range(round(1.0 * a.fps)):  # хвост, чтобы последняя реплика успела закрыться
        writer.write(np.zeros((h, w, 3), np.uint8))
    writer.release()

    for row, st in zip(timeline, starts):
        row['start_s'] = round(st / a.fps, 2)
        row['end_s'] = round((st + row['frames']) / a.fps, 2)
    manifest = {'video': str(a.out), 'fps': a.fps, 'frames': total,
                'turns': turns, 'clips': timeline,
                'note': 'видео собрано из тестовых клипов Slovo; в репозиторий не кладётся'}
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                encoding='utf-8')
    print(f'{a.out}: {total} кадров, {len(clips)} клипов, манифест {a.manifest}', flush=True)


if __name__ == '__main__':
    main()
