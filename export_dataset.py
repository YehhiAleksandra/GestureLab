#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Экспорт сессий GestureLab в датасет этапа 1: окна 1-2 с по 15 к/с,
векторы точек (2 кисти × 63), метки жестов. Сплит — по исполнителям
(train/val по людям, не по повторам): защита от утечки между сетами.

Пример: python export_dataset.py sessions --out dataset.npz --window 1.5 --stride 0.75
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import session_schema
from gl_session import KIND_GESTURE, kind_of
from summarize import load_sessions

FEATURE_DIM = 126  # 2 кисти × 63 координаты


def frame_vector(fr) -> list:
    """Признак кадра: кисть 1 + кисть 2; отсутствующая кисть — нули."""
    lms = fr["hand_landmarks"]
    v = list(lms[0]) if len(lms) >= 1 else [0.0] * 63
    v += list(lms[1]) if len(lms) >= 2 else [0.0] * 63
    return v


def resample(frames, fps: float):
    """Кадры сессии -> равномерная сетка fps по t_ms; пропуск — None (окно бракуется)."""
    if not frames:
        return []
    ts = np.array([fr["t_ms"] for fr in frames], float)
    vecs = [frame_vector(fr) for fr in frames]
    n = int(ts[-1] * fps / 1000.0) + 1
    grid = np.arange(n) * 1000.0 / fps
    idx = np.abs(ts[None, :] - grid[:, None]).argmin(axis=1)
    ok = np.abs(ts[idx] - grid) <= 500.0 / fps
    return [vecs[i] if good else None for i, good in zip(idx, ok)]


def windows(vecs, fps: float, window: float, stride: float):
    t = int(round(window * fps))
    step = max(1, int(round(stride * fps)))
    for start in range(0, max(1, len(vecs) - t + 1), step):
        win = vecs[start:start + t]
        if len(win) == t and all(v is not None for v in win):
            yield np.array(win, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="папка с JSON-сессиями")
    ap.add_argument("--out", default="dataset.npz")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--window", type=float, default=1.5, help="длина окна, с (1-2 по ТЗ)")
    ap.add_argument("--stride", type=float, default=0.75, help="шаг окна, с")
    ap.add_argument("--val-persons", nargs="*", help="исполнители в val (по умолчанию каждый второй)")
    args = ap.parse_args()

    sessions, n_found, _ = load_sessions(args.folder)
    good = []
    for d in sessions:
        if kind_of(d) != KIND_GESTURE or session_schema.validate_session(d):
            continue
        good.append(d)
    if not good:
        sys.exit(f"В {args.folder} нет валидных сессий-жестов.")

    names = sorted({d["gesture"] for d in good})
    label = {g: i for i, g in enumerate(names)}
    X, y, person, sess_id = [], [], [], []
    for si, d in enumerate(good):
        vecs = resample(d["frames"], args.fps)
        for win in windows(vecs, args.fps, args.window, args.stride):
            X.append(win)
            y.append(label[d["gesture"]])
            person.append(d["person"])
            sess_id.append(si)
    if not X:
        sys.exit("Ни одного окна не собрано: сессии короче окна или без обеих кистей.")

    X = np.array(X, np.float32)
    y = np.array(y, np.int64)
    person = np.array(person, np.object_)
    sess_id = np.array(sess_id, np.int64)

    persons = sorted(set(person))
    if args.val_persons:
        val_persons = [p for p in args.val_persons if p in persons]
    elif len(persons) >= 2:
        val_persons = persons[1::2]
    else:
        val_persons = []
        print("ВНИМАНИЕ: исполнитель один — сплит по людям невозможен, "
              "val собран по сессиям (риск утечки исполнителя).")
    is_val = np.isin(person, val_persons) if val_persons else (sess_id % 2 == 1)

    # np.savez дописывает .npz, если расширения нет: печатаем реальное имя файла (B24)
    dest = Path(args.out)
    if dest.suffix != ".npz":
        dest = dest.with_suffix(".npz")
    np.savez(dest, X=X, y=y, person=person, is_val=is_val,
             names=np.array(names, np.object_), fps=args.fps, window=args.window)
    print(f"окон: {len(X)} ({args.window} с @ {args.fps} к/с) | классов: {len(names)}")
    for g in names:
        print(f"  {g:24s} {int((y == label[g]).sum()):5d}")
    print(f"train: {int((~is_val).sum())} окон, люди {sorted(set(person[~is_val]))}")
    print(f"val:   {int(is_val.sum())} окон, люди {sorted(set(person[is_val])) or '(по сессиям)'}")
    print(f"датасет: {dest}")


if __name__ == "__main__":
    main()
