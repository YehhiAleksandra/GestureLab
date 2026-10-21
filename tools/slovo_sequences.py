"""slovo_mediapipe.json (1.26 ГБ) → slovo_seq.npz: все кадры каждого видео + официальный train/test.

Читает JSON потоково (stdlib raw_decode), чтобы не держать файл целиком в памяти.
Руки в кадре упорядочены по x запястья (левая на изображении — первая): тот же
порядок обязан применять живой режим, иначе признаки разойдутся.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

SRC = Path('datasets/slovo/slovo_mediapipe.json')
ANN = Path('datasets/slovo/annotations.csv')
OUT = Path('datasets/slovo/slovo_seq.npz')
CHUNK = 1 << 22
KEY = re.compile(r'\s*,?\s*"([^"]+)"\s*:\s*')


def iter_videos(path):
    dec = json.JSONDecoder()
    with open(path, encoding='utf-8') as f:
        buf = f.read(CHUNK)
        pos = buf.index('{') + 1
        while True:
            m = KEY.match(buf, pos)
            while True:
                try:
                    if m is None:
                        raise ValueError
                    value, end = dec.raw_decode(buf, m.end())
                    break
                except ValueError:  # JSONDecodeError тоже ValueError: значение оборвано блоком
                    more = f.read(CHUNK)
                    if not more:
                        return
                    buf, pos = buf[pos:] + more, 0
                    m = KEY.match(buf, pos)
            yield m.group(1), value
            pos = end
            if pos > CHUNK:
                buf, pos = buf[pos:], 0


def frame_array(frame):
    hands = []
    for name in ('hand 1', 'hand 2'):
        pts = frame.get(name) if isinstance(frame, dict) else None
        if pts and len(pts) == 21:
            hands.append([[p['x'], p['y'], p['z']] for p in pts])
    hands.sort(key=lambda h: h[0][0])
    out = np.full((2, 21, 3), np.nan, dtype=np.float32)
    for i, h in enumerate(hands[:2]):
        out[i] = h
    return out


def main():
    with open(ANN, encoding='utf-8') as f:
        ann = {r['attachment_id']: r for r in csv.DictReader(f, delimiter='\t')}

    frames, offsets, ids, texts, users, train = [], [0], [], [], [], []
    for n, (vid, vframes) in enumerate(iter_videos(SRC), 1):
        a = ann.get(vid)
        if a is None or not vframes:
            continue
        arr = np.stack([frame_array(fr) for fr in vframes]).astype(np.float16)
        frames.append(arr)
        offsets.append(offsets[-1] + len(arr))
        ids.append(vid)
        texts.append(a['text'])
        users.append(a['user_id'])
        train.append(a['train'] == 'True')
        if n % 2000 == 0:
            print(f'{n} видео, кадров {offsets[-1]}', flush=True)

    np.savez(OUT, frames=np.concatenate(frames), offsets=np.array(offsets, dtype=np.int64),
             ids=np.array(ids), texts=np.array(texts), users=np.array(users),
             train=np.array(train))
    tr = np.array(train)
    print(f'Готово: {len(ids)} видео ({tr.sum()} train / {(~tr).sum()} test), '
          f'{offsets[-1]} кадров, {len(set(texts))} классов → {OUT}')
    print(f'Людей: train {len(set(np.array(users)[tr]))}, test {len(set(np.array(users)[~tr]))}, '
          f'пересечение {len(set(np.array(users)[tr]) & set(np.array(users)[~tr]))}')


if __name__ == '__main__':
    main()
