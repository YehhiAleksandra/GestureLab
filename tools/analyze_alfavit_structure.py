#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ структуры видео alfavit_landmarks.json.
Показывает распределение кадров с руками и помогает определить таймкоды букв.
"""

import json
import sys
from pathlib import Path


def analyze_landmarks(json_path: str):
    """Анализировать распределение рук в видео."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fps = data["fps"]
    frames = data["frames"]
    total = len(frames)

    print(f"Всего кадров: {total}")
    print(f"FPS: {fps}")
    print(f"Длительность: {total/fps:.1f} сек ({total/fps/60:.1f} мин)")
    print()

    # Подсчёт кадров с руками
    with_hands = [f for f in frames if f["hands"]]
    print(f"Кадров с руками: {len(with_hands)} ({len(with_hands)/total*100:.1f}%)")
    print()

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
                    "duration": current_count / fps,
                    "frames": current_count
                })
                current_start = None
                current_count = 0

    # Добавить последний сегмент, если видео закончилось с руками
    if current_start is not None:
        segments.append({
            "start": current_start,
            "end": current_start + current_count - 1,
            "duration": current_count / fps,
            "frames": current_count
        })

    print(f"Непрерывных сегментов с руками: {len(segments)}")
    print()

    # Показать первые 20 сегментов
    print("Первые 20 сегментов:")
    for i, seg in enumerate(segments[:20]):
        start_time = seg["start"] / fps
        end_time = seg["end"] / fps
        print(f"  {i+1}. {start_time:.1f}s - {end_time:.1f}s ({seg['duration']:.1f}s, {seg['frames']} кадров)")

    print()
    print(f"Всего сегментов: {len(segments)}")

    # Статистика по длительности сегментов
    if segments:
        durations = [s["duration"] for s in segments]
        print(f"Средняя длительность сегмента: {sum(durations)/len(durations):.2f}s")
        print(f"Мин: {min(durations):.2f}s, Макс: {max(durations):.2f}s")


if __name__ == "__main__":
    analyze_landmarks("datasets/youtube_ark/alfavit_landmarks.json")
