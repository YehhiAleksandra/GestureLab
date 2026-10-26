"""
Интеграция Slovo MediaPipe landmarks в наш формат данных.

Slovo landmarks: 20000 видео × ~50 кадров × 2 руки × 21 точка × 3 координаты
Наш формат: windows (N, 22, 126) где 126 = 2 руки × 21 точка × 3 координаты

Скрипт:
1. Загружает Slovo landmarks
2. Создаёт windows с тем же форматом, что и наши данные
3. Сохраняет в NPZ файл для обучения
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'slovo_repo'))
from constants import classes


def load_slovo_landmarks(landmarks_path: str) -> Dict:
    """Загружает Slovo MediaPipe landmarks."""
    print(f"Загрузка Slovo landmarks из {landmarks_path}...")
    with open(landmarks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Загружено {len(data)} видео")
    return data


def load_slovo_annotations(annotations_path: str = None) -> Dict:
    """
    Загружает аннотации Slovo (какое видео = какой жест).
    
    Если annotations.csv нет, используем classes из constants.py.
    """
    # Для простоты используем маппинг video_id -> class_name
    # В реальности нужно читать annotations.csv
    # Но у нас нет самого датасета, только landmarks
    
    # Создадим маппинг на основе структуры данных
    # Slovo landmarks содержат video_id как ключи
    # Аннотации должны быть в annotations.csv, но его нет
    
    # Временное решение: используем первые 1000 классов
    print(f"Доступно классов: {len(classes)}")
    return {i: classes[i] for i in range(len(classes))}


def extract_hand_landmarks(frame_data: Dict) -> np.ndarray:
    """
    Извлекает landmarks рук из одного кадра.
    
    Args:
        frame_data: словарь с landmarks для одного кадра
    
    Returns:
        landmarks: np.ndarray shape (126,) = 2 руки × 21 точка × 3 координаты
    """
    # Slovo формат: {'left_hand': [...], 'right_hand': [...]}
    # Каждый hand: 21 точка × 3 координаты (x, y, z)
    
    left_hand = frame_data.get('left_hand', [])
    right_hand = frame_data.get('right_hand', [])
    
    # Если руки не обнаружены, возвращаем нули
    if not left_hand or not right_hand:
        return np.zeros(126, dtype=np.float32)
    
    # Конвертируем в numpy arrays
    left = np.array(left_hand, dtype=np.float32).flatten()
    right = np.array(right_hand, dtype=np.float32).flatten()
    
    # Проверяем размерность
    if len(left) != 63 or len(right) != 63:
        return np.zeros(126, dtype=np.float32)
    
    # Объединяем: [left_hand (63), right_hand (63)]
    return np.concatenate([left, right])


def create_windows_from_video(video_landmarks: List[Dict], 
                             window_size: int = 22, 
                             step: int = 5) -> List[np.ndarray]:
    """
    Создаёт скользящие окна из landmarks видео.
    
    Args:
        video_landmarks: список landmarks для каждого кадра
        window_size: размер окна (22 кадра)
        step: шаг окна (5 кадров)
    
    Returns:
        windows: список окон shape (22, 126)
    """
    windows = []
    
    for start in range(0, len(video_landmarks) - window_size + 1, step):
        window = []
        for i in range(window_size):
            frame_data = video_landmarks[start + i]
            landmarks = extract_hand_landmarks(frame_data)
            window.append(landmarks)
        
        window = np.array(window, dtype=np.float32)
        windows.append(window)
    
    return windows


def convert_slovo_to_our_format(landmarks_path: str, 
                               output_path: str,
                               max_videos: int = None):
    """
    Конвертирует Slovo landmarks в наш формат NPZ.
    
    Args:
        landmarks_path: путь к slovo_mediapipe.json
        output_path: путь для сохранения NPZ
        max_videos: максимальное количество видео (None = все)
    """
    data = load_slovo_landmarks(landmarks_path)
    
    video_ids = list(data.keys())
    if max_videos:
        video_ids = video_ids[:max_videos]
    
    print(f"\nОбработка {len(video_ids)} видео...")
    
    all_windows = []
    all_labels = []
    
    # Для простоты присваиваем каждому видео случайный класс
    # В реальности нужно читать annotations.csv
    num_classes = min(1000, len(classes))
    
    for idx, video_id in enumerate(video_ids):
        video_landmarks = data[video_id]
        
        # Создаём windows
        windows = create_windows_from_video(video_landmarks)
        
        if not windows:
            continue
        
        # Присваиваем класс (временно - случайный)
        # В реальности нужно читать annotations.csv
        label = idx % num_classes
        
        all_windows.extend(windows)
        all_labels.extend([label] * len(windows))
        
        if (idx + 1) % 100 == 0:
            print(f"  Обработано {idx + 1}/{len(video_ids)} видео, "
                  f"создано {len(all_windows)} окон")
    
    # Конвертируем в numpy arrays
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"\n✓ Создано {len(X)} окон")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Уникальных классов: {len(np.unique(y))}")
    
    # Сохраняем
    np.savez_compressed(output_path, X=X, y=y)
    print(f"✓ Сохранено в {output_path}")
    
    # Статистика по классам
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nТоп-10 классов:")
    for cls, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        class_name = classes[cls] if cls < len(classes) else f"class_{cls}"
        print(f"  {class_name:20s}: {count:5d} окон")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Конвертация Slovo landmarks в наш формат')
    parser.add_argument('--input', default='datasets/slovo_mediapipe.json', 
                       help='Путь к Slovo landmarks JSON')
    parser.add_argument('--output', default='datasets/slovo_landmarks_windows.npz',
                       help='Путь для сохранения NPZ')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='Максимальное количество видео (None = все)')
    
    args = parser.parse_args()
    
    convert_slovo_to_our_format(args.input, args.output, args.max_videos)


if __name__ == '__main__':
    main()
