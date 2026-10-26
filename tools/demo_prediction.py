#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демо-визуализация работы модели на видео из Bukva.
Показывает кадры с предсказаниями модели и правильными ответами.

Пример: python tools/demo_prediction.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import load_model, normalize


def resample_landmarks(landmarks_list, target_fps=15):
    """Ресемплинг ландмарков в target_fps Hz."""
    if len(landmarks_list) == 0:
        return np.array([])
    
    n_frames = len(landmarks_list)
    duration = n_frames / 30.0  # исходные видео 30 FPS
    target_n = int(duration * target_fps)
    
    if target_n == 0:
        return np.array([])
    
    indices = np.linspace(0, n_frames - 1, target_n)
    resampled = [landmarks_list[int(i)] for i in indices]
    return np.array(resampled)


def process_video(video_path, model, names):
    """Обработать видео и вернуть предсказания для каждого окна."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    frames = []
    landmarks_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            points = []
            for l in lm.landmark:
                points.extend([l.x, l.y, l.z])
            # Дублировать для второй руки (модель обучалась на 126 признаках)
            landmarks_list.append(points + points)
        else:
            landmarks_list.append(None)
    
    cap.release()
    hands.close()
    
    if len(frames) == 0:
        return None, None, None
    
    # Ресемплинг в 15 Hz
    valid_landmarks = [l for l in landmarks_list if l is not None]
    if len(valid_landmarks) < 22:
        return frames, [], []
    
    resampled = resample_landmarks(valid_landmarks, target_fps=15)
    if len(resampled) < 22:
        return frames, [], []
    
    # Нарезка на окна по 22 кадра
    window_size = 22
    stride = 11
    predictions = []
    confidences = []
    
    for i in range(0, len(resampled) - window_size + 1, stride):
        window = resampled[i:i + window_size]
        X = window.reshape(1, window_size, -1).astype(np.float32)
        X_norm = normalize(X)
        prob, _ = model.forward(X_norm)
        pred_idx = np.argmax(prob[0])
        predictions.append(names[pred_idx])
        confidences.append(float(prob[0][pred_idx]))
    
    return frames, predictions, confidences


def visualize_prediction(frames, predictions, confidences, output_dir, video_name, gt_label):
    """Сохранить кадры с предсказаниями."""
    if len(frames) == 0 or len(predictions) == 0:
        return
    
    # Выбрать 5 кадров равномерно по видео
    n_frames = len(frames)
    indices = np.linspace(0, n_frames - 1, min(5, n_frames)).astype(int)
    
    for i, frame_idx in enumerate(indices):
        frame = frames[frame_idx].copy()
        h, w = frame.shape[:2]
        
        # Добавить текст с предсказанием
        if i < len(predictions):
            pred = predictions[i]
            conf = confidences[i]
            text = f"Предсказание: {pred} ({conf:.1%})"
        else:
            text = "Обработка..."
        
        # Фон для текста
        cv2.rectangle(frame, (10, 10), (w - 10, 80), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        
        # Правильный ответ
        gt_text = f"Правильно: {gt_label}"
        cv2.putText(frame, gt_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
        
        # Сохранить
        output_path = output_dir / f"{video_name}_frame_{i}.png"
        cv2.imwrite(str(output_path), frame)
        print(f"Сохранён кадр: {output_path}")


def main():
    # Загрузить модель
    model_path = Path("bukva_model_full.npz")
    if not model_path.exists():
        print("Модель не найдена: bukva_model_full.npz")
        return
    
    model, names = load_model(model_path)
    print(f"Модель загружена: {len(names)} классов")
    
    # Найти видео с правильными ответами
    annotations_path = Path("datasets/bukva/annotations.tsv")
    if not annotations_path.exists():
        print("annotations.tsv не найден")
        return
    
    # Прочитать разметку
    import csv
    video_labels = {}
    with open(annotations_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            video_id = row['attachment_id']
            label = row['text']
            video_labels[video_id] = label
    
    # Выбрать случайное видео
    trimmed_dir = Path("datasets/bukva/trimmed")
    videos = list(trimmed_dir.glob("*.mp4"))
    
    if len(videos) == 0:
        print("Нет видео в datasets/bukva/trimmed/")
        return
    
    # Взять первые 3 видео для демо
    demo_videos = videos[:3]
    
    output_dir = Path("reports/img/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_path in demo_videos:
        video_name = video_path.stem
        gt_label = video_labels.get(video_name, "???")
        
        print(f"\nОбработка: {video_name} (правильно: {gt_label})")
        
        frames, predictions, confidences = process_video(video_path, model, names)
        
        if frames is None:
            print(f"  Не удалось открыть видео")
            continue
        
        if len(predictions) == 0:
            print(f"  Не удалось извлечь ландмарки")
            continue
        
        print(f"  Кадров: {len(frames)}, предсказаний: {len(predictions)}")
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            print(f"  Окно {i}: {pred} ({conf:.1%})")
        
        visualize_prediction(frames, predictions, confidences, output_dir, video_name, gt_label)
    
    print(f"\nДемо сохранено в: {output_dir}")


if __name__ == "__main__":
    main()
