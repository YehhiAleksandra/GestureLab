#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработка датасета Slovo через готовые MediaPipe landmarks.
Slovo: 20400 видео, 1000 классов РЖЯ (включая дактильную азбуку).

Использует готовые landmarks из slovo_mediapipe.json вместо обработки видео.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_slovo_landmarks(landmarks_path: Path) -> dict:
    """Загрузить готовые MediaPipe landmarks из JSON."""
    print(f"Загрузка landmarks из {landmarks_path}...")
    with open(landmarks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Загружено {len(data)} записей")
    return data


def load_annotations(annotations_path: Path) -> list:
    """Загрузить аннотации Slovo."""
    rows = []
    with open(annotations_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    print(f"Загружено {len(rows)} аннотаций")
    return rows


def resample_landmarks(landmarks_list, target_n=22):
    """Ресемплинг ландмарков к фиксированному числу кадров."""
    if len(landmarks_list) == 0:
        return None
    if len(landmarks_list) == target_n:
        return np.array(landmarks_list)
    indices = np.linspace(0, len(landmarks_list) - 1, target_n)
    resampled = [landmarks_list[int(i)] for i in indices]
    return np.array(resampled)


def process_slovo(landmarks_data: dict, annotations: list, output_path: Path):
    """Обработать Slovo и сохранить в формате для train.py."""
    # Собрать метки классов
    labels = sorted(set(row['text'] for row in annotations))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    print(f"Классов: {len(labels)}")

    X_windows = []
    y_labels = []
    is_val = []

    # Разметка val: каждое 5-е видео
    val_indices = set(range(0, len(annotations), 5))

    for i, row in enumerate(annotations):
        video_id = row['attachment_id']
        label = row['text']
        is_val_flag = i in val_indices

        if video_id not in landmarks_data:
            continue

        video_landmarks = landmarks_data[video_id]
        if len(video_landmarks) == 0:
            continue

        # Извлечь точки рук (21 точка × 3 координаты = 63 на руку)
        # Slovo использует одну руку для дактильной азбуки
        frames_landmarks = []
        for frame in video_landmarks:
            if 'hand 1' in frame:
                hand = frame['hand 1']
                points = []
                for lm in hand:
                    points.extend([lm['x'], lm['y'], lm['z']])
                # Дублировать для второй руки (как в Bukva)
                frames_landmarks.append(points + points)
            else:
                frames_landmarks.append(None)

        # Убрать None
        valid = [f for f in frames_landmarks if f is not None]
        if len(valid) < 22:
            continue

        # Ресемплинг к 22 кадрам
        resampled = resample_landmarks(valid, target_n=22)
        if resampled is None:
            continue

        X_windows.append(resampled)
        y_labels.append(label_to_idx[label])
        is_val.append(is_val_flag)

    if len(X_windows) == 0:
        print("Нет окон для обучения!")
        return

    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    is_val = np.array(is_val, dtype=bool)

    print(f"Окон: {len(X_windows)}")
    print(f"Форма X: {X.shape}")
    print(f"Распределение по классам: min={np.bincount(y).min()}, max={np.bincount(y).max()}")

    np.savez(output_path, X=X, y=y, is_val=is_val, names=np.array(labels))
    print(f"Сохранено в {output_path}")


def main():
    slovo_dir = Path("datasets/slovo")
    landmarks_path = slovo_dir / "slovo_mediapipe.json"
    annotations_path = slovo_dir / "annotations.csv"
    output_path = slovo_dir / "slovo_processed.npz"

    if not landmarks_path.exists():
        print(f"Landmarks не найдены: {landmarks_path}")
        print("Скачайте: curl -L -o slovo_mediapipe.json https://...")
        sys.exit(1)

    if not annotations_path.exists():
        print(f"Аннотации не найдены: {annotations_path}")
        sys.exit(1)

    landmarks_data = load_slovo_landmarks(landmarks_path)
    annotations = load_annotations(annotations_path)
    process_slovo(landmarks_data, annotations, output_path)


if __name__ == "__main__":
    main()
