"""
Извлечение реальных MediaPipe landmarks для CSLR сегментов.
Для каждого сегмента из cslr_dataset_from_easy_sign.json извлекает landmarks из видео.
"""
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_landmarks_for_segment(landmarks_data: dict, start_sec: float, end_sec: float, fps: float, target_frames: int = 100):
    """
    Извлекает landmarks для сегмента [start_sec, end_sec].
    Возвращает [target_frames, 126] — 2 руки × 21 точка × 3 координаты.
    """
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    frames = landmarks_data['frames']
    
    # Извлекаем кадры сегмента
    segment_frames = []
    for frame_idx in range(start_frame, min(end_frame, len(frames))):
        frame = frames[frame_idx]
        
        # Извлекаем landmarks рук
        hands = frame.get('hands', [])
        if len(hands) == 0:
            # Нет рук — пропускаем
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
        return None
    
    # Равномерно сэмплируем target_frames кадров
    if len(segment_frames) >= target_frames:
        indices = np.linspace(0, len(segment_frames) - 1, target_frames, dtype=int)
        segment_frames = [segment_frames[i] for i in indices]
    else:
        # Дополняем последним кадром
        while len(segment_frames) < target_frames:
            segment_frames.append(segment_frames[-1])
    
    return np.array(segment_frames, dtype=np.float32)


def main():
    # Загружаем датасет с pseudo-labels
    dataset_path = Path('datasets/youtube_rsl/cslr_dataset_from_easy_sign.json')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    segments = dataset['segments']
    print(f"Загружено {len(segments)} сегментов")
    
    # Загружаем landmarks для каждого видео
    landmarks_files = {
        '26RsAcQCVBA': Path('datasets/youtube_rsl/26RsAcQCVBA_landmarks.json'),
        '9UdEIyOCQUY': Path('datasets/youtube_rsl/9UdEIyOCQUY_landmarks.json'),
        'Qcc3VBT4VT4': Path('datasets/youtube_rsl/Qcc3VBT4VT4_landmarks.json'),
    }
    
    landmarks_data = {}
    fps_map = {}
    for video_id, path in landmarks_files.items():
        print(f"Загружаю landmarks: {path.name}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        landmarks_data[video_id] = data
        fps_map[video_id] = data['fps']
        print(f"  Кадров: {len(data['frames'])}, FPS: {data['fps']:.2f}")
    
    # Извлекаем landmarks для каждого сегмента
    print(f"\nИзвлечение landmarks для {len(segments)} сегментов...")
    
    segments_with_landmarks = []
    skipped = 0
    
    for seg in tqdm(segments, desc="Извлечение"):
        video_id = seg['video_id']
        start_sec = seg['start_sec']
        end_sec = seg['end_sec']
        fps = fps_map[video_id]
        
        landmarks = extract_landmarks_for_segment(
            landmarks_data[video_id],
            start_sec,
            end_sec,
            fps,
            target_frames=100
        )
        
        if landmarks is None:
            skipped += 1
            continue
        
        segments_with_landmarks.append({
            'video_id': video_id,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'gesture': seg['gesture'],
            'gesture_id': seg['gesture_id'],
            'confidence': seg['confidence'],
            'features': landmarks.tolist(),  # [100, 126]
        })
    
    print(f"\n=== Итого ===")
    print(f"Обработано: {len(segments)}")
    print(f"Пропущено (нет рук): {skipped}")
    print(f"С landmarks: {len(segments_with_landmarks)}")
    
    # Сохраняем
    output_path = Path('datasets/youtube_rsl/cslr_dataset_with_real_landmarks.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_segments': len(segments_with_landmarks),
            'feature_dim': 126,
            'target_frames': 100,
            'segments': segments_with_landmarks,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nДатасет сохранён: {output_path}")
    
    # Статистика
    gesture_counts = {}
    for seg in segments_with_landmarks:
        g = seg['gesture']
        gesture_counts[g] = gesture_counts.get(g, 0) + 1
    
    print(f"\nУникальных жестов: {len(gesture_counts)}")
    print("Топ-10:")
    for g, c in sorted(gesture_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {g}: {c}")


if __name__ == '__main__':
    main()
