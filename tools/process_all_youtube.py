#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработка всех YouTube видео через MediaPipe.
Извлекает ландмарки рук из каждого кадра.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def process_video(video_path: str, output_path: str, sample_rate: int = 5):
    """Обработать видео через MediaPipe."""
    print(f"Обработка: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть {video_path}", file=sys.stderr)
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps}, Кадров: {total_frames}")
    
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frames_data = []
    frame_idx = 0
    processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Пропускаем кадры для ускорения
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        # Конвертируем в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                hands_data.append(landmarks)
        
        frames_data.append({
            "frame": frame_idx,
            "timestamp": frame_idx / fps,
            "hands": hands_data
        })
        
        processed += 1
        if processed % 100 == 0:
            print(f"  Обработано: {processed}/{total_frames // sample_rate} кадров")
        
        frame_idx += 1
    
    cap.release()
    mp_hands.close()
    
    print(f"  Всего обработано: {processed} кадров")
    print(f"  Кадров с руками: {sum(1 for f in frames_data if f['hands'])}")
    
    # Сохраняем
    output = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "sample_rate": sample_rate,
        "frames": frames_data
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    
    print(f"  Сохранено: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Обработка YouTube видео")
    parser.add_argument("--input-dir", default="datasets/youtube", help="Директория с видео")
    parser.add_argument("--output-dir", default="datasets/youtube_processed", help="Директория для JSON")
    parser.add_argument("--sample-rate", type=int, default=5, help="Обрабатывать каждый N-й кадр")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Находим все MP4 файлы
    video_files = list(input_dir.rglob("*.mp4"))
    print(f"Найдено видео: {len(video_files)}")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] {video_path.name}")
        output_path = output_dir / f"{video_path.stem}_landmarks.json"
        
        if output_path.exists():
            print(f"  Уже обработано, пропускаем")
            continue
        
        process_video(str(video_path), str(output_path), args.sample_rate)
    
    print(f"\nГотово! Обработано {len(video_files)} видео")


if __name__ == "__main__":
    main()
