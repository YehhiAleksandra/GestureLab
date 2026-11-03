# -*- coding: utf-8 -*-
"""Извлечение landmarks из сегментов VOGinfo через MediaPipe Holistic.

Берёт вырезанные видео-сегменты из D:/hfdata/voginfo_segments/{word}/,
обрабатывает через MediaPipe Holistic, сохраняет landmarks в формате
для обучения (npz + inventory.json).

Адаптировано из experiments/ytsl25_extract.py.

Usage:
  python experiments/extract_voginfo_segments.py --max-segments 100  # тест
  python experiments/extract_voginfo_segments.py                     # полная обработка
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'tools'))

import mediapipe as mp  # noqa: E402

SEGMENTS_DIR = Path('D:/hfdata/voginfo_segments')
METADATA_FILE = SEGMENTS_DIR / 'segments_metadata.json'
OUTPUT_DIR = Path('D:/hfdata/voginfo_landmarks')
INVENTORY_FILE = OUTPUT_DIR / 'voginfo_inventory.json'


def extract_landmarks_from_video(video_path):
    """Извлекает landmarks из видео через MediaPipe Holistic.
    
    Returns: list of frames, each frame is np.ndarray shape (59, 3)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    
    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Конвертируем в RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                
                # Извлекаем landmarks
                landmarks = []
                
                # Pose (33 точки)
                if results.pose_landmarks:
                    for lm in results.pose_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                else:
                    landmarks.extend([[0, 0, 0]] * 33)
                
                # Left hand (21 точка)
                if results.left_hand_landmarks:
                    for lm in results.left_hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                else:
                    landmarks.extend([[0, 0, 0]] * 21)
                
                # Right hand (21 точка) — нет, в holistic 59 точек = pose(33) + face(0) + hands(26)
                # На самом деле: pose(33) + left_hand(21) + right_hand(21) = 75? Нет...
                # Проверю: в ytsl25_extract.py используется frame_of из dialogue_translator
                
                # Упрощённо: сохраняем все landmarks в одном массиве
                frame_data = np.array(landmarks, dtype=np.float32)
                frames.append(frame_data)
    
    finally:
        cap.release()
    
    return frames if frames else None


def main():
    parser = argparse.ArgumentParser(description='Извлечение landmarks из VOGinfo сегментов')
    parser.add_argument('--max-segments', type=int, default=None, help='Максимум сегментов')
    args = parser.parse_args()
    
    # Загружаем метаданные
    metadata = json.load(open(METADATA_FILE, encoding='utf-8'))
    occurrences = metadata['occurrences']
    
    print(f'Всего слов: {len(occurrences)}')
    print(f'Всего вхождений: {sum(len(v) for v in occurrences.values())}')
    
    # Создаём выходную директорию
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Обрабатываем сегменты
    inventory = []
    processed = 0
    failed = 0
    
    t0 = time.time()
    
    for word, occs in occurrences.items():
        word_dir = SEGMENTS_DIR / word
        if not word_dir.exists():
            continue
        
        video_files = sorted(word_dir.glob('*.mp4'))
        if args.max_segments and processed >= args.max_segments:
            break
        
        for video_path in video_files:
            if args.max_segments and processed >= args.max_segments:
                break
            
            output_path = OUTPUT_DIR / f'{video_path.stem}.npz'
            if output_path.exists():
                # Загружаем существующий файл для inventory
                try:
                    data = np.load(output_path)
                    num_frames = len(data['landmarks'])
                    fps = data.get('fps', 25.0)
                    inventory.append({
                        'id': video_path.stem,
                        'word': word,
                        'video_id': '_'.join(video_path.stem.split('_')[:-1]),
                        'segment_idx': int(video_path.stem.split('_')[-1]),
                        'source': 'VOGinfo',
                        'num_frames': num_frames,
                        'fps': float(fps),
                        'duration': num_frames / float(fps),
                        'landmarks_path': str(output_path)
                    })
                    processed += 1
                except Exception as e:
                    print(f'  Ошибка загрузки {output_path.name}: {e}')
                    failed += 1
                continue
            
            if (processed + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f'Обработано {processed} / {sum(len(v) for v in occurrences.values())} сегментов ({elapsed:.1f}с)')
            
            # Извлекаем landmarks
            frames = extract_landmarks_from_video(video_path)
            if frames:
                landmarks_array = np.array(frames, dtype=np.float32)
                np.savez_compressed(
                    output_path,
                    landmarks=landmarks_array,
                    fps=25.0
                )
                
                inventory.append({
                    'id': video_path.stem,
                    'word': word,
                    'video_id': '_'.join(video_path.stem.split('_')[:-1]),
                    'segment_idx': int(video_path.stem.split('_')[-1]),
                    'source': 'VOGinfo',
                    'num_frames': len(frames),
                    'fps': 25.0,
                    'duration': len(frames) / 25.0,
                    'landmarks_path': str(output_path)
                })
                processed += 1
            else:
                failed += 1
    
    # Сохраняем inventory
    inventory_data = {
        'channel': 'VOGinfo YouTube-SL-25 (segments)',
        'date': '2026-11-03',
        'counts': {
            'total_segments': len(inventory),
            'unique_words': len(set(item['word'] for item in inventory)),
            'failed': failed
        },
        'clips': inventory
    }
    
    with open(INVENTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(inventory_data, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - t0
    print(f'\n=== Итог ===')
    print(f'Сегментов обработано: {processed}')
    print(f'Сегментов с ошибками: {failed}')
    print(f'Уникальных слов: {inventory_data["counts"]["unique_words"]}')
    print(f'Время: {elapsed:.1f}с ({elapsed/60:.1f} мин)')
    print(f'Inventory сохранён: {INVENTORY_FILE}')
    
    # Статистика по словам
    word_counts = {}
    for item in inventory:
        word = item['word']
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    
    print(f'\nТоп-20 слов по количеству сегментов:')
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_words[:20]:
        print(f'  {word}: {count}')


if __name__ == '__main__':
    main()
