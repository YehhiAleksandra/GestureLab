# -*- coding: utf-8 -*-
"""Извлечение landmarks из сегментов VOGinfo через MediaPipe Holistic.

Берёт вырезанные видео-сегменты из D:/hfdata/voginfo_segments/{word}/,
обрабатывает через MediaPipe Holistic, сохраняет landmarks в формате
для обучения (npz + inventory.json).

Usage:
  python experiments/extract_voginfo_landmarks.py --max-segments 100  # тест
  python experiments/extract_voginfo_landmarks.py                     # полная обработка
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.sign_stream import extract_holistic  # noqa: E402

SEGMENTS_DIR = Path('D:/hfdata/voginfo_segments')
METADATA_FILE = SEGMENTS_DIR / 'segments_metadata.json'
OUTPUT_DIR = Path('D:/hfdata/voginfo_landmarks')
INVENTORY_FILE = OUTPUT_DIR / 'voginfo_inventory.json'


def process_segment(video_path, output_path):
    """Обрабатывает один видео-сегмент, сохраняет landmarks."""
    try:
        result = extract_holistic(str(video_path))
        if result is None or len(result['landmarks']) == 0:
            return None
        
        # Сохраняем landmarks
        np.savez_compressed(
            output_path,
            landmarks=result['landmarks'],
            handedness=result.get('handedness', []),
            fps=result.get('fps', 30.0)
        )
        
        return {
            'num_frames': len(result['landmarks']),
            'fps': result.get('fps', 30.0),
            'duration': len(result['landmarks']) / result.get('fps', 30.0)
        }
    except Exception as e:
        print(f'  Ошибка обработки {video_path.name}: {e}')
        return None


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
                processed += 1
                continue
            
            if (processed + 1) % 50 == 0:
                print(f'Обработано {processed} / {sum(len(v) for v in occurrences.values())} сегментов')
            
            result = process_segment(video_path, output_path)
            if result:
                inventory.append({
                    'id': video_path.stem,
                    'word': word,
                    'video_id': '_'.join(video_path.stem.split('_')[:-1]),
                    'segment_idx': int(video_path.stem.split('_')[-1]),
                    'source': 'VOGinfo',
                    'num_frames': result['num_frames'],
                    'fps': result['fps'],
                    'duration': result['duration'],
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
    
    print(f'\n=== Итог ===')
    print(f'Сегментов обработано: {processed}')
    print(f'Сегментов с ошибками: {failed}')
    print(f'Уникальных слов: {inventory_data["counts"]["unique_words"]}')
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
