# -*- coding: utf-8 -*-
"""Конвертация VOGinfo landmarks в формат SP-10 (59 точек).

VOGinfo сохранены как (N, 54, 3): pose(33) + hands(21).
SP-10 формат: (N, 59, 3): hands(42) + pose(7) + face(10).

Переоцениваем видео через MediaPipe Holistic с правильной функцией frame_of.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import mediapipe as mp
from dialogue_translator import frame_of

VOGINFO_SEGMENTS = Path('D:/hfdata/voginfo_segments')
OUTPUT_DIR = Path('D:/hfdata/voginfo_landmarks_fixed')
INVENTORY_FILE = OUTPUT_DIR / 'voginfo_inventory.json'


def convert_video(video_path, output_path):
    """Переоценить видео и сохранить в формате (N, 59, 3)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    
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
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                landmarks = frame_of(results)  # (59, 3)
                frames.append(landmarks)
    finally:
        cap.release()
    
    if not frames:
        return None, None
    
    frames_array = np.array(frames, dtype=np.float32)
    np.savez_compressed(output_path, frames=frames_array, fps=fps)
    return len(frames), fps


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Загружаем метаданные VOGinfo
    metadata_file = VOGINFO_SEGMENTS / 'segments_metadata.json'
    metadata = json.load(open(metadata_file, encoding='utf-8'))
    occurrences = metadata['occurrences']
    
    print(f'Всего слов: {len(occurrences)}')
    print(f'Всего вхождений: {sum(len(v) for v in occurrences.values())}')
    
    inventory = []
    processed = 0
    failed = 0
    
    for word, occs in occurrences.items():
        word_dir = VOGINFO_SEGMENTS / word
        if not word_dir.exists():
            continue
        
        video_files = sorted(word_dir.glob('*.mp4'))
        
        for video_path in video_files:
            output_path = OUTPUT_DIR / f'{video_path.stem}.npz'
            
            if output_path.exists():
                # Уже конвертирован
                try:
                    data = np.load(output_path)
                    num_frames = len(data['frames'])
                    fps = float(data['fps'])
                    inventory.append({
                        'id': video_path.stem,
                        'word': word,
                        'video_id': '_'.join(video_path.stem.rsplit('_', 1)[0].split('_')[-2:]) if '_' in video_path.stem else video_path.stem,
                        'segment_idx': int(video_path.stem.rsplit('_', 1)[-1]) if '_' in video_path.stem else 1,
                        'source': 'VOGinfo',
                        'num_frames': num_frames,
                        'fps': fps,
                        'duration': num_frames / fps,
                        'landmarks_path': str(output_path)
                    })
                    processed += 1
                except Exception as e:
                    print(f'  Ошибка загрузки {output_path.name}: {e}')
                    failed += 1
                continue
            
            # Конвертируем
            num_frames, fps = convert_video(video_path, output_path)
            if num_frames is None:
                failed += 1
                continue
            
            inventory.append({
                'id': video_path.stem,
                'word': word,
                'video_id': '_'.join(video_path.stem.rsplit('_', 1)[0].split('_')[-2:]) if '_' in video_path.stem else video_path.stem,
                'segment_idx': int(video_path.stem.rsplit('_', 1)[-1]) if '_' in video_path.stem else 1,
                'source': 'VOGinfo',
                'num_frames': num_frames,
                'fps': float(fps),
                'duration': num_frames / float(fps),
                'landmarks_path': str(output_path)
            })
            processed += 1
            
            if processed % 50 == 0:
                print(f'  Обработано {processed} видео...')
    
    # Сохраняем inventory
    inv_data = {
        'channel': 'VOGinfo YouTube-SL-25 (segments, fixed format)',
        'date': '2026-11-03',
        'counts': {
            'total_segments': len(inventory),
            'unique_words': len(set(c['word'] for c in inventory)),
            'failed': failed
        },
        'clips': inventory
    }
    
    with open(INVENTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(inv_data, f, ensure_ascii=False, indent=2)
    
    print(f'\n✓ Конвертация завершена')
    print(f'  Успешно: {processed}')
    print(f'  Ошибок: {failed}')
    print(f'  Inventory: {INVENTORY_FILE}')


if __name__ == '__main__':
    main()
