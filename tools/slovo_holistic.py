"""karinversan/slovo-full-landmarks (HF, MediaPipe Holistic, 68 parquet-блоков) → slovo_holistic.npz.

Каждый блок качается, ужимается до точек из slovo_model.POSE_IDX/FACE_IDX и кэшируется
отдельно, поэтому прерванный прогон продолжается с места остановки. Кадры — только
отрезок жеста begin..end, как их отдаёт датасет.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live_signs import FACE_IDX, POSE_IDX, holistic_frame  # noqa: E402

REPO = 'karinversan/slovo-full-landmarks'
COLS = ['video_id', 'gloss', 'signer_id', 'split', 'hand_0', 'hand_0_mask', 'hand_1',
        'hand_1_mask', 'pose', 'pose_mask', 'face', 'face_mask']


def points(col, n):
    return col.flatten().flatten().flatten().to_numpy(zero_copy_only=False).reshape(-1, n, 3)


def convert(parquet):
    frames, lens, meta = [], [], []
    for b in pq.ParquetFile(parquet).iter_batches(batch_size=50, columns=COLS):
        lens_b = b.column('hand_0').value_lengths().to_numpy()
        h0, h1 = points(b.column('hand_0'), 21), points(b.column('hand_1'), 21)
        pose = points(b.column('pose'), 33)[:, POSE_IDX]
        face = points(b.column('face'), 478)[:, FACE_IDX]
        m = {k: b.column(k + '_mask').flatten().to_numpy(zero_copy_only=False)
             for k in ('hand_0', 'hand_1', 'pose', 'face')}
        frames.append(np.stack([holistic_frame(h0[i] if m['hand_0'][i] else None,
                                               h1[i] if m['hand_1'][i] else None,
                                               pose[i] if m['pose'][i] else None,
                                               face[i] if m['face'][i] else None)
                                for i in range(len(h0))]).astype(np.float16))
        lens.append(lens_b)
        meta += zip(*(b.column(k).to_pylist() for k in ('video_id', 'gloss', 'signer_id', 'split')))
    return np.concatenate(frames), np.concatenate(lens), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=r'D:\hfdata\slovo_full')
    ap.add_argument('--out', default='datasets/slovo/slovo_holistic.npz')
    a = ap.parse_args()
    cache = Path(a.cache)
    (cache / 'compact').mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type='dataset') if f.startswith('data/'))

    for n, f in enumerate(files, 1):
        dst = cache / 'compact' / (f.replace('/', '_') + '.npz')
        if dst.exists():
            continue
        src = hf_hub_download(REPO, f, repo_type='dataset', local_dir=cache)
        frames, lens, meta = convert(src)
        np.savez(dst, frames=frames, lens=lens, meta=np.array(meta))
        Path(src).unlink()
        print(f'{n}/{len(files)} {f}: {len(lens)} видео, {len(frames)} кадров', flush=True)

    frames, offsets, meta = [], [0], []
    for f in files:
        d = np.load(cache / 'compact' / (f.replace('/', '_') + '.npz'))
        frames.append(d['frames'])
        offsets += list(offsets[-1] + np.cumsum(d['lens']))
        meta.append(d['meta'])
    meta = np.concatenate(meta)
    np.savez(a.out, frames=np.concatenate(frames), offsets=np.array(offsets, dtype=np.int64),
             ids=meta[:, 0], texts=meta[:, 1], users=meta[:, 2], train=meta[:, 3] == 'train')
    print(f'Готово: {len(meta)} видео, {offsets[-1]} кадров → {a.out}')


if __name__ == '__main__':
    main()
