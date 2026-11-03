# -*- coding: utf-8 -*-
"""Быстрая конвертация VOGinfo landmarks: (N, 54, 3) → (N, 59, 3).

Старый формат: pose(33) + left_hand(21) = 54 точки.
Новый формат: left_hand(21) + right_hand(21) + pose(7) + face(10) = 59 точек.

Поскольку в старых данных нет правой руки и лица, заполняем NaN.
"""
import json
from pathlib import Path

import numpy as np

OLD_DIR = Path('D:/hfdata/voginfo_landmarks')
NEW_DIR = Path('D:/hfdata/voginfo_landmarks_converted')
INVENTORY_FILE = NEW_DIR / 'voginfo_inventory.json'


def convert_npz(old_path, new_path):
    """Конвертировать один NPZ файл."""
    data = np.load(old_path)
    old_landmarks = data['landmarks']  # (N, 54, 3)
    fps = float(data.get('fps', 25.0))
    
    N = len(old_landmarks)
    new_landmarks = np.full((N, 59, 3), np.nan, dtype=np.float32)
    
    # Старый формат: pose(0:33) + left_hand(33:54)
    # Новый формат: left_hand(0:21) + right_hand(21:42) + pose(42:49) + face(49:59)
    
    # Копируем левую руку из старого формата в новый
    new_landmarks[:, 0:21] = old_landmarks[:, 33:54]
    
    # Копируем pose (только 7 точек из 33)
    # В старом формате pose - все 33 точки, в новом - только 7 ключевых
    # Для простоты копируем первые 7
    new_landmarks[:, 42:49] = old_landmarks[:, 0:7]
    
    # Правая рука и лицо остаются NaN
    
    np.savez_compressed(new_path, frames=new_landmarks, fps=fps)
    return N, fps


def main():
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Загружаем старый inventory для метаданных
    old_inv = json.load(open(OLD_DIR / 'voginfo_inventory.json', encoding='utf-8'))
    
    inventory = []
    processed = 0
    failed = 0
    
    for clip in old_inv['clips']:
        old_path = OLD_DIR / f"{clip['id']}.npz"
        new_path = NEW_DIR / f"{clip['id']}.npz"
        
        if not old_path.exists():
            failed += 1
            continue
        
        if new_path.exists():
            # Уже конвертирован
            try:
                data = np.load(new_path)
                num_frames = len(data['frames'])
                fps = float(data['fps'])
                inventory.append({
                    'id': clip['id'],
                    'word': clip['word'],
                    'video_id': clip['video_id'],
                    'segment_idx': clip['segment_idx'],
                    'source': 'VOGinfo',
                    'num_frames': num_frames,
                    'fps': fps,
                    'duration': num_frames / fps,
                    'landmarks_path': str(new_path)
                })
                processed += 1
            except Exception as e:
                print(f'  Ошибка загрузки {new_path.name}: {e}')
                failed += 1
            continue
        
        try:
            num_frames, fps = convert_npz(old_path, new_path)
            inventory.append({
                'id': clip['id'],
                'word': clip['word'],
                'video_id': clip['video_id'],
                'segment_idx': clip['segment_idx'],
                'source': 'VOGinfo',
                'num_frames': num_frames,
                'fps': fps,
                'duration': num_frames / fps,
                'landmarks_path': str(new_path)
            })
            processed += 1
            
            if processed % 100 == 0:
                print(f'  Конвертировано {processed} файлов...')
        except Exception as e:
            print(f'  Ошибка конвертации {old_path.name}: {e}')
            failed += 1
    
    # Сохраняем inventory
    inv_data = {
        'channel': 'VOGinfo YouTube-SL-25 (converted format)',
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
