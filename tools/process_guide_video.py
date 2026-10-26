#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработка видео guide_dagestan.mp4 через MediaPipe Hands.
Извлекает ландмарки рук из видео с сурдопереводом.
"""

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp


def extract_landmarks(video_path: str, output_json: str):
    """Извлечь ландмарки рук из видео."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не могу открыть {video_path}", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Видео: {video_path}")
    print(f"FPS: {fps}, кадров: {total_frames}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frames_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_landmarks = {
            "frame": frame_idx,
            "timestamp": frame_idx / fps if fps > 0 else 0,
            "hands": []
        }

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = []
                for lm in hand_landmarks.landmark:
                    points.append([lm.x, lm.y, lm.z])
                frame_landmarks["hands"].append(points)

        frames_data.append(frame_landmarks)
        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"Обработано {frame_idx}/{total_frames} кадров", file=sys.stderr)

    cap.release()
    hands.close()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "frames": frames_data
        }, f, ensure_ascii=False, indent=2)

    print(f"Сохранено: {output_json}")
    print(f"Кадров с руками: {sum(1 for f in frames_data if f['hands'])}/{len(frames_data)}")


if __name__ == "__main__":
    video = "datasets/youtube_ark/holy_sepulchre.mp4"
    output = "datasets/youtube_ark/holy_sepulchre_landmarks.json"
    extract_landmarks(video, output)
