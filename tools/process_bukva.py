#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработка датасета Bukva (дактильная азбука РЖЯ) через MediaPipe.
Читает видео из trimmed/, извлекает точки ландмарков кистей, сохраняет в формате для train.py.

Формат выхода: bukva_processed.npz
- X: (n_windows, 22, 126) — окна по 1.5 сек при 15 Hz, 126 признаков (2 руки × 21 точка × 3 координаты)
- y: (n_windows,) — метки классов (0-32 для 33 букв)
- is_val: (n_windows,) — булев массив (True для валидации)
- names: список имён классов (33 буквы)

Пример: python tools/process_bukva.py datasets/bukva --out bukva_processed.npz
"""

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def load_annotations(annotations_path):
    """Читает annotations.tsv, возвращает список словарей."""
    rows = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({
                "attachment_id": row["attachment_id"],
                "user_id": row["user_id"],
                "text": row["text"].strip(),
                "begin": int(row["begin"]),
                "end": int(row["end"]),
                "train": row["train"].strip() == "True",
            })
    return rows


def process_video(video_path, hands):
    """
    Обрабатывает видео через MediaPipe Hands.
    Возвращает список кадров: каждый кадр — массив (126,) для рук или None если рук нет.
    Для дактильной азбуки достаточно одной руки.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1:
            landmarks = np.zeros(126, dtype=np.float32)
            # Берём первую руку (для дактильной азбуки обычно одна рука)
            hand_landmarks = results.multi_hand_landmarks[0]
            for point_idx, point in enumerate(hand_landmarks.landmark):
                landmarks[point_idx * 3] = point.x
                landmarks[point_idx * 3 + 1] = point.y
                landmarks[point_idx * 3 + 2] = point.z
            frames.append(landmarks)
        else:
            frames.append(None)

    cap.release()
    return frames, fps


def resample_frames(frames, fps, target_fps=15.0):
    """Ресемплирует кадры в target_fps Hz по nearest."""
    if not frames or fps <= 0:
        return []

    ratio = target_fps / fps
    n_target = max(1, int(len(frames) * ratio))
    resampled = []

    for i in range(n_target):
        src_idx = min(int(i / ratio), len(frames) - 1)
        resampled.append(frames[src_idx])

    return resampled


def extract_windows(frames, window_sec=1.5, stride_sec=0.75, fps=15.0):
    """
    Нарезает ресемплированные кадры на окна.
    Окно: window_sec * fps = 22 кадра при 15 Hz.
    Stride: stride_sec * fps = 11 кадров при 15 Hz.
    """
    window_size = int(window_sec * fps)
    stride = int(stride_sec * fps)

    windows = []
    for start in range(0, len(frames) - window_size + 1, stride):
        window = frames[start:start + window_size]
        if all(f is not None for f in window):
            windows.append(np.array(window))

    return windows


def main():
    ap = argparse.ArgumentParser(description="Обработка Bukva через MediaPipe")
    ap.add_argument("bukva_dir", help="Путь к распакованному датасету Bukva")
    ap.add_argument("--out", default="bukva_processed.npz", help="Выходной .npz файл")
    ap.add_argument("--max-videos", type=int, default=0, help="Максимум видео для обработки (0 = все)")
    args = ap.parse_args()

    bukva_dir = Path(args.bukva_dir)
    annotations_path = bukva_dir / "annotations.tsv"
    trimmed_dir = bukva_dir / "trimmed"

    if not annotations_path.exists():
        print(f"Ошибка: {annotations_path} не найден")
        return

    if not trimmed_dir.exists():
        print(f"Ошибка: {trimmed_dir} не найдена")
        return

    print(f"Читаю annotations.tsv...")
    annotations = load_annotations(annotations_path)
    print(f"Найдено {len(annotations)} видео")

    # Собираем список уникальных букв
    letters = sorted(set(a["text"] for a in annotations))
    letter_to_idx = {letter: i for i, letter in enumerate(letters)}
    print(f"Классы ({len(letters)}): {' '.join(letters)}")

    # Инициализируем MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_windows = []
    all_labels = []
    all_is_val = []

    videos_to_process = annotations[:args.max_videos] if args.max_videos > 0 else annotations

    for i, ann in enumerate(videos_to_process):
        video_name = ann["attachment_id"]
        video_path = trimmed_dir / f"{video_name}.mp4"
        if not video_path.exists():
            print(f"[{i+1}/{len(videos_to_process)}] Видео не найдено: {video_path.name}")
            continue

        print(f"[{i+1}/{len(videos_to_process)}] Обработка: {video_path.name} (буква '{ann['text']}')")

        frames, fps = process_video(video_path, hands)
        if not frames:
            print(f"  Пропуск: не удалось открыть видео")
            continue

        resampled = resample_frames(frames, fps)
        windows = extract_windows(resampled)

        if not windows:
            print(f"  Пропуск: нет окон с детектированной рукой")
            continue

        label = letter_to_idx[ann["text"]]
        is_val = not ann["train"]

        for window in windows:
            all_windows.append(window)
            all_labels.append(label)
            all_is_val.append(is_val)

        print(f"  Извлечено {len(windows)} окон")

    hands.close()

    if not all_windows:
        print("Ошибка: не извлечено ни одного окна")
        return

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    is_val = np.array(all_is_val, dtype=bool)
    names = np.array(letters)

    out_path = Path(args.out)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")

    np.savez(out_path, X=X, y=y, is_val=is_val, names=names)
    print(f"\nГотово! Сохранено: {out_path}")
    print(f"Всего окон: {len(X)} | Классов: {len(letters)} | Валидация: {is_val.sum()}")


if __name__ == "__main__":
    main()
