#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Извлечение окон букв из видео alfavit_landmarks.json.
Каждый сегмент с руками = одна буква алфавита.
"""

import json
import sys
from pathlib import Path

import numpy as np


# Русский алфавит (33 буквы)
RUSSIAN_ALPHABET = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"


def extract_windows(json_path: str, output_npz: str, window_size: int = 22):
    """Извлечь окна жестов из ландмарков."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fps = data["fps"]
    frames = data["frames"]

    # Найти непрерывные сегменты с руками
    segments = []
    current_start = None
    current_count = 0

    for f in frames:
        if f["hands"]:
            if current_start is None:
                current_start = f["frame"]
                current_count = 1
            else:
                current_count += 1
        else:
            if current_start is not None:
                segments.append({
                    "start": current_start,
                    "end": current_start + current_count - 1,
                    "frames": current_count
                })
                current_start = None
                current_count = 0

    if current_start is not None:
        segments.append({
            "start": current_start,
            "end": current_start + current_count - 1,
            "frames": current_count
        })

    print(f"Найдено сегментов: {len(segments)}")
    print(f"Букв в алфавите: {len(RUSSIAN_ALPHABET)}")

    # Присваиваем буквы сегментам по порядку
    # Предполагаем: первые 33 сегмента = 33 буквы (может быть с повторами)
    windows = []
    labels = []

    for i, seg in enumerate(segments):
        # Определяем букву для этого сегмента
        letter_idx = i % len(RUSSIAN_ALPHABET)
        letter = RUSSIAN_ALPHABET[letter_idx]

        # Извлекаем кадры сегмента
        start_frame = seg["start"]
        end_frame = seg["end"]
        seg_frames = frames[start_frame:end_frame + 1]

        # Фильтруем только кадры с руками
        frames_with_hands = [f for f in seg_frames if f["hands"]]

        if len(frames_with_hands) < window_size:
            # Пропускаем слишком короткие сегменты
            continue

        # Ресемплируем до window_size кадров
        indices = np.linspace(0, len(frames_with_hands) - 1, window_size, dtype=int)
        resampled = [frames_with_hands[idx] for idx in indices]

        # Извлекаем ландмарки
        landmarks_window = []
        for frame in resampled:
            # Берём первую руку (если есть)
            if frame["hands"]:
                hand = frame["hands"][0]
                # 21 точка × 3 координаты = 63 features
                points = []
                for point in hand:
                    points.extend(point)
                # Дублируем для второй руки (модель ожидает 126 features)
                landmarks_window.append(points + points)

        if len(landmarks_window) == window_size:
            windows.append(landmarks_window)
            labels.append(letter)

    print(f"Извлечено окон: {len(windows)}")

    # Подсчёт по буквам
    from collections import Counter
    label_counts = Counter(labels)
    print(f"Распределение по буквам:")
    for letter in sorted(label_counts.keys()):
        print(f"  {letter}: {label_counts[letter]}")

    # Сохраняем
    np.savez(
        output_npz,
        windows=np.array(windows, dtype=np.float32),
        labels=np.array(labels),
        alphabet=np.array(list(RUSSIAN_ALPHABET))
    )
    print(f"Сохранено: {output_npz}")


if __name__ == "__main__":
    extract_windows(
        "datasets/youtube_ark/alfavit_landmarks.json",
        "datasets/youtube_ark/alfavit_windows.npz"
    )
