#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Предсказание букв на видео с сурдопереводом и озвучка через VoiceStudio.
"""

import json
import sys
from pathlib import Path

import numpy as np
import requests

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))
from train import GRU, normalize


def load_model(model_path: str):
    """Загрузить обученную модель."""
    data = np.load(model_path, allow_pickle=True)
    names = data["names"]
    idx_to_label = {i: name for i, name in enumerate(names)}
    
    hidden_dim = int(data["hidden"])
    
    model = GRU(
        d=126,
        h=hidden_dim,
        c=len(names),
        seed=0
    )
    
    # Восстанавливаем веса
    for key in ["Wz", "bz", "Wr", "br", "Wh", "bh", "Wo", "bo"]:
        model.p[key] = data[key]
    
    return model, idx_to_label


def predict_letter(model, idx_to_label, landmarks_window):
    """Предсказать букву по окну ландмарков."""
    X = np.array([landmarks_window], dtype=np.float32)
    X_norm = normalize(X)
    prob, _ = model.forward(X_norm)
    pred_idx = np.argmax(prob[0])
    return idx_to_label[pred_idx], float(prob[0][pred_idx])


def extract_windows_from_landmarks(landmarks_path: str, window_size: int = 22):
    """Извлечь окна жестов из ландмарков видео."""
    with open(landmarks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
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
    
    # Извлекаем окна из сегментов
    windows = []
    for seg in segments:
        start_frame = seg["start"]
        end_frame = seg["end"]
        seg_frames = frames[start_frame:end_frame + 1]
        
        # Фильтруем только кадры с руками
        frames_with_hands = [f for f in seg_frames if f["hands"]]
        
        if len(frames_with_hands) < window_size:
            continue
        
        # Ресемплируем до window_size кадров
        indices = np.linspace(0, len(frames_with_hands) - 1, window_size, dtype=int)
        resampled = [frames_with_hands[idx] for idx in indices]
        
        # Извлекаем ландмарки
        landmarks_window = []
        for frame in resampled:
            if frame["hands"]:
                hand = frame["hands"][0]
                points = []
                for point in hand:
                    points.extend(point)
                # Дублируем для второй руки
                landmarks_window.append(points + points)
        
        if len(landmarks_window) == window_size:
            windows.append({
                "landmarks": landmarks_window,
                "start_time": seg["start"] / data["fps"],
                "end_time": seg["end"] / data["fps"]
            })
    
    return windows


def synthesize_speech(text: str, output_path: str):
    """Озвучить текст через VoiceStudio TTS."""
    url = "http://localhost:3900/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": "564b8c9f",  # Клон Александра
        "response_format": "mp3"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        print(f"Озвучка сохранена: {output_path}")
        return True
    except Exception as e:
        print(f"Ошибка озвучки: {e}", file=sys.stderr)
        return False


def main():
    landmarks_path = "datasets/youtube_ark/holy_sepulchre_landmarks.json"
    model_path = "bukva_model_combined.npz"
    output_text = "datasets/youtube_ark/holy_sepulchre_text.txt"
    output_audio = "datasets/youtube_ark/holy_sepulchre_audio.mp3"
    
    if not Path(landmarks_path).exists():
        print(f"Ошибка: {landmarks_path} не найден", file=sys.stderr)
        print("Сначала обработайте видео через process_guide_video.py", file=sys.stderr)
        return
    
    print("Загрузка модели...")
    model, idx_to_label = load_model(model_path)
    
    print("Извлечение окон из видео...")
    windows = extract_windows_from_landmarks(landmarks_path)
    print(f"Найдено {len(windows)} окон с жестами")
    
    print("Предсказание букв...")
    predictions = []
    for i, window in enumerate(windows):
        letter, confidence = predict_letter(model, idx_to_label, window["landmarks"])
        predictions.append({
            "letter": letter,
            "confidence": confidence,
            "start_time": window["start_time"],
            "end_time": window["end_time"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Обработано {i + 1}/{len(windows)} окон")
    
    # Собираем текст
    text = "".join(p["letter"] for p in predictions)
    print(f"\nПредсказанный текст: {text[:100]}...")
    
    # Сохраняем текст
    with open(output_text, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Текст сохранён: {output_text}")
    
    # Озвучиваем
    print("\nОзвучка текста...")
    synthesize_speech(text, output_audio)


if __name__ == "__main__":
    main()
