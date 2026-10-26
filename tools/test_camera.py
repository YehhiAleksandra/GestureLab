#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест обученной модели Bukva на реальной камере в реальном времени.
Показывает предсказанную букву и отправляет в VoiceStudio для озвучки.

Пример: python tools/test_camera.py bukva_model.npz
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests

from train import load_model, normalize


def predict_letter(model, names, landmarks_window):
    """Предсказать букву по окну ландмарков."""
    X = np.array([landmarks_window], dtype=np.float32)
    X_norm = normalize(X)
    prob, _ = model.forward(X_norm)
    pred_idx = np.argmax(prob[0])
    return names[pred_idx], float(prob[0][pred_idx])


def speak_text(text, voice_api_url="http://localhost:3900/v1/audio/speech"):
    """Отправить текст в VoiceStudio для озвучки."""
    try:
        response = requests.post(
            voice_api_url,
            json={
                "model": "tts-1",
                "input": text,
                "voice": "Aleksandra",
                "response_format": "mp3",
            },
            timeout=5,
        )
        if response.status_code == 200:
            # Сохранить аудио и воспроизвести
            audio_path = Path("temp_speech.mp3")
            audio_path.write_bytes(response.content)
            # Воспроизведение через системный плеер (Windows)
            import subprocess
            subprocess.Popen(["start", str(audio_path)], shell=True)
            return True
    except Exception as e:
        print(f"Ошибка озвучки: {e}")
    return False


def main():
    ap = argparse.ArgumentParser(description="Тест модели на камере")
    ap.add_argument("model", help="Путь к обученной модели (.npz)")
    ap.add_argument("--speak", action="store_true", help="Озвучивать буквы через VoiceStudio")
    args = ap.parse_args()

    # Загрузить модель
    print(f"Загрузка модели: {args.model}")
    model, names = load_model(args.model)
    print(f"Классы: {len(names)}")

    # Инициализировать MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Открыть камеру
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    print("Камера открыта. Показывайте букву дактильной азбуки!")
    print("Нажмите 'q' для выхода")

    landmarks_buffer = []
    last_prediction = None
    last_prediction_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.zeros(63, dtype=np.float32)
            for i, point in enumerate(hand_landmarks.landmark):
                landmarks[i * 3] = point.x
                landmarks[i * 3 + 1] = point.y
                landmarks[i * 3 + 2] = point.z

            landmarks_buffer.append(landmarks)

            # Нужно 22 кадра для окна
            if len(landmarks_buffer) >= 22:
                window = np.array(landmarks_buffer[-22:])
                # Расширить до 126 признаков (2 руки, но у нас одна)
                window_126 = np.zeros((22, 126), dtype=np.float32)
                window_126[:, :63] = window

                letter, confidence = predict_letter(model, names, window_126)

                # Показывать предсказание не чаще раза в секунду
                current_time = time.time()
                if letter != last_prediction or current_time - last_prediction_time > 1.0:
                    print(f"Буква: {letter} (уверенность: {confidence:.2f})")
                    last_prediction = letter
                    last_prediction_time = current_time

                    if args.speak:
                        speak_text(letter)

                # Оставить последние 11 кадров для следующего окна (stride 11)
                landmarks_buffer = landmarks_buffer[-11:]

            # Нарисовать руку
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

        cv2.imshow("Bukva Test (press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
