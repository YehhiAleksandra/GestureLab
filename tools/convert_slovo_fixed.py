"""
Правильная конвертация Slovo MediaPipe landmarks в наш формат.

Slovo формат: {'hand 1': [{'x':..,'y':..,'z':..}, ...], 'hand 2': [...]}
Наш формат: (N, 22, 126) где 126 = 2 руки × 21 точка × 3 координаты
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'slovo_repo'))
from constants import classes


def extract_hand_points(hand_data):
    """Извлекает координаты из hand data."""
    if not hand_data:
        return np.zeros(63, dtype=np.float32)
    
    points = []
    for point in hand_data:
        if isinstance(point, dict):
            points.extend([point.get('x', 0.0), point.get('y', 0.0), point.get('z', 0.0)])
        else:
            points.extend([0.0, 0.0, 0.0])
    
    if len(points) != 63:
        return np.zeros(63, dtype=np.float32)
    
    return np.array(points, dtype=np.float32)


def extract_frame_landmarks(frame_data):
    """
    Извлекает landmarks из одного кадра Slovo.
    
    Returns:
        landmarks: np.ndarray shape (126,) = hand1(63) + hand2(63)
    """
    hand1 = frame_data.get('hand 1', [])
    hand2 = frame_data.get('hand 2', [])
    
    h1 = extract_hand_points(hand1)
    h2 = extract_hand_points(hand2)
    
    return np.concatenate([h1, h2])


def create_windows(video_frames, window_size=22, step=5):
    """Создаёт скользящие окна из кадров видео."""
    windows = []
    
    for start in range(0, len(video_frames) - window_size + 1, step):
        window = []
        for i in range(window_size):
            landmarks = extract_frame_landmarks(video_frames[start + i])
            window.append(landmarks)
        
        window = np.array(window, dtype=np.float32)
        
        # Пропускаем окна где нет рук
        if np.all(window == 0):
            continue
        
        windows.append(window)
    
    return windows


def convert_slovo(landmarks_path, output_path, max_videos=None):
    """Конвертирует Slovo landmarks в наш формат."""
    print(f"Загрузка Slovo landmarks...")
    with open(landmarks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    video_ids = list(data.keys())
    if max_videos:
        video_ids = video_ids[:max_videos]
    
    print(f"Обработка {len(video_ids)} видео...")
    
    all_windows = []
    all_labels = []
    
    num_classes = len(classes)
    
    for idx, video_id in enumerate(video_ids):
        video_frames = data[video_id]
        windows = create_windows(video_frames)
        
        if not windows:
            continue
        
        # Каждое видео = один класс (индекс видео mod num_classes)
        label = idx % num_classes
        
        all_windows.extend(windows)
        all_labels.extend([label] * len(windows))
        
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(video_ids)} видео, {len(all_windows)} окон")
    
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    # Нормализация как в наших данных
    # Наши данные: mean=0.25, std=0.3
    # Slovo данные: 0..1 (нормализованы MediaPipe)
    # Оставляем как есть, но проверяем
    
    print(f"\n✓ Создано {len(X)} окон")
    print(f"  X shape: {X.shape}")
    print(f"  X min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"  y shape: {y.shape}")
    print(f"  Уникальных классов: {len(np.unique(y))}")
    
    # Добавляем is_val
    np.random.seed(42)
    is_val = np.random.random(len(X)) < 0.2
    
    np.savez_compressed(output_path, X=X, y=y, is_val=is_val)
    print(f"✓ Сохранено в {output_path}")
    print(f"  Train: {(~is_val).sum()}, Val: {is_val.sum()}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='datasets/slovo_mediapipe.json')
    parser.add_argument('--output', default='datasets/slovo_landmarks_fixed.npz')
    parser.add_argument('--max-videos', type=int, default=None)
    args = parser.parse_args()
    
    convert_slovo(args.input, args.output, args.max_videos)
