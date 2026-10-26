"""
Извлечение сегментов из Bible RSL видео для CSLR обучения.
Использует weak labels для создания датасета.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict


def extract_segments(
    landmarks_path: Path,
    weak_labels: List[Dict],
    video_id: str,
    output_dir: Path
):
    """
    Извлекает сегменты из landmarks JSON согласно weak labels.
    """
    # Загружаем landmarks
    print(f"Загружаю landmarks: {landmarks_path}")
    with open(landmarks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fps = data['fps']
    frames = data['frames']
    
    # Фильтруем labels для этого видео
    video_labels = [l for l in weak_labels if l['video_id'] == video_id]
    print(f"Найдено {len(video_labels)} сегментов для {video_id}")
    
    segments = []
    
    for label in video_labels:
        start_frame = int(label['start_sec'] * fps)
        end_frame = int(label['end_sec'] * fps)
        
        # Извлекаем кадры сегмента
        segment_frames = []
        for frame_idx in range(start_frame, min(end_frame, len(frames))):
            frame = frames[frame_idx]
            
            # Извлекаем landmarks рук
            hands = frame.get('hands', [])
            if len(hands) == 0:
                # Нет рук — пропускаем или используем нулевые landmarks
                continue
            
            # Берём первую руку (или обе, если есть)
            hand_landmarks = []
            for hand in hands[:2]:  # Максимум 2 руки
                # 21 точка × 3 координаты (x, y, z) = 63 значения на руку
                flat_landmarks = []
                for point in hand:
                    flat_landmarks.extend(point)
                hand_landmarks.append(flat_landmarks)
            
            # Если одна рука, дублируем для второй (или используем zeros)
            if len(hand_landmarks) == 1:
                hand_landmarks.append([0.0] * 63)
            elif len(hand_landmarks) == 0:
                hand_landmarks = [[0.0] * 63, [0.0] * 63]
            
            # Конкатенируем: 63 + 63 = 126 признаков
            frame_features = hand_landmarks[0] + hand_landmarks[1]
            segment_frames.append(frame_features)
        
        if len(segment_frames) == 0:
            print(f"  Пропускаю {label['reference']}: нет landmarks")
            continue
        
        # Конвертируем в numpy array
        segment_array = np.array(segment_frames, dtype=np.float32)
        
        segment = {
            'video_id': video_id,
            'reference': label['reference'],
            'label': label['label'],
            'start_sec': label['start_sec'],
            'end_sec': label['end_sec'],
            'num_frames': len(segment_frames),
            'features': segment_array.tolist(),  # [num_frames, 126]
        }
        segments.append(segment)
        
        print(f"  {label['reference']}: {len(segment_frames)} кадров, shape {segment_array.shape}")
    
    return segments


def main():
    # Загружаем weak labels
    weak_labels_path = Path('datasets/youtube_rsl/weak_labels.json')
    with open(weak_labels_path, 'r', encoding='utf-8') as f:
        weak_labels = json.load(f)
    
    print(f"Загружено {len(weak_labels)} weak labels")
    
    # Обрабатываем каждое видео
    all_segments = []
    
    videos = [
        ('26RsAcQCVBA', 'datasets/youtube_rsl/26RsAcQCVBA_landmarks.json'),
        ('9UdEIyOCQUY', 'datasets/youtube_rsl/9UdEIyOCQUY_landmarks.json'),
        ('Qcc3VBT4VT4', 'datasets/youtube_rsl/Qcc3VBT4VT4_landmarks.json'),
    ]
    
    for video_id, landmarks_path in videos:
        print(f"\n=== Обработка {video_id} ===")
        segments = extract_segments(
            Path(landmarks_path),
            weak_labels,
            video_id,
            Path('datasets/youtube_rsl/segments')
        )
        all_segments.extend(segments)
    
    # Сохраняем датасет
    output_path = Path('datasets/youtube_rsl/cslr_dataset.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_segments': len(all_segments),
            'feature_dim': 126,  # 2 руки × 21 точка × 3 координаты
            'segments': all_segments
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Итого ===")
    print(f"Создано {len(all_segments)} сегментов")
    print(f"Датасет сохранён: {output_path}")
    
    # Статистика по labels
    label_counts = {}
    for seg in all_segments:
        label = seg['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nУникальных labels: {len(label_counts)}")
    print("Примеры:")
    for label, count in list(label_counts.items())[:10]:
        print(f"  {label}: {count}")


if __name__ == '__main__':
    main()
