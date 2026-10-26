#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработка YouTube видео через MediaPipe для обучения модели.
Извлекает landmarks и добавляет в датасет.
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import normalize


def process_video(video_path: Path, output_path: Path, label: str, label_to_idx: dict):
    """Обработать одно видео и сохранить landmarks."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Не удалось открыть: {video_path}")
        return None

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    frames_landmarks = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            points = []
            for l in lm.landmark:
                points.extend([l.x, l.y, l.z])
            # Дублировать для второй руки (как в Bukva)
            frames_landmarks.append(points + points)
        else:
            frames_landmarks.append(None)

    cap.release()
    hands.close()

    if len(frames_landmarks) == 0:
        return None

    # Убрать None
    valid = [f for f in frames_landmarks if f is not None]
    if len(valid) < 22:
        return None

    # Ресемплинг к 22 кадрам (15 Hz)
    duration = len(frames_landmarks) / 30.0
    target_n = int(duration * 15)
    if target_n < 22:
        return None

    indices = np.linspace(0, len(valid) - 1, target_n)
    resampled = [valid[int(i)] for i in indices]

    # Нарезка на окна
    windows = []
    window_size = 22
    stride = 11
    for i in range(0, len(resampled) - window_size + 1, stride):
        window = np.array(resampled[i:i + window_size], dtype=np.float32)
        windows.append(window)

    if len(windows) == 0:
        return None

    # Сохранить
    label_idx = label_to_idx.get(label, -1)
    if label_idx == -1:
        return None

    data = {
        "video": str(video_path.name),
        "label": label,
        "label_idx": label_idx,
        "windows": len(windows),
        "landmarks": [w.tolist() for w in windows]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    return data


def main():
    video_dir = Path("datasets/youtube_rsl")
    output_dir = video_dir / "processed"
    output_dir.mkdir(exist_ok=True)

    # Маппинг букв (дактильная азбука)
    dactyl_letters = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    label_to_idx = {letter: i for i, letter in enumerate(dactyl_letters)}
    label_to_idx["no_event"] = len(dactyl_letters)

    # Найти все видео
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
    print(f"Найдено видео: {len(videos)}")

    all_windows = []
    all_labels = []

    for video_path in videos:
        # Извлечь букву из имени файла (если есть)
        name = video_path.stem
        label = None
        for letter in dactyl_letters:
            if letter in name:
                label = letter
                break

        if label is None:
            print(f"Пропуск (нет буквы в имени): {video_path.name}")
            continue

        print(f"Обработка: {video_path.name} (буква: {label})")
        output_path = output_dir / f"{video_path.stem}.json"
        result = process_video(video_path, output_path, label, label_to_idx)

        if result:
            for window in result["landmarks"]:
                all_windows.append(window)
                all_labels.append(result["label_idx"])
            print(f"  Добавлено окон: {result['windows']}")
        else:
            print(f"  Не удалось обработать")

    if len(all_windows) == 0:
        print("Нет данных для обучения!")
        return

    # Сохранить в формате train.py
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    # Разметка val: каждое 5-е окно
    is_val = np.array([i % 5 == 0 for i in range(len(X))], dtype=bool)

    output_npz = video_dir / "youtube_dataset.npz"
    names = np.array(list(dactyl_letters) + ["no_event"])
    np.savez(output_npz, X=X, y=y, is_val=is_val, names=names)

    print(f"\nСохранено: {output_npz}")
    print(f"Окон: {len(X)}")
    print(f"Классов: {len(names)}")


if __name__ == "__main__":
    main()
