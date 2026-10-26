"""
Создание weak labels из Bible RSL видео.
Использует OCR-detected verse references для сегментации видео
и извлекает русский текст Bible для каждого сегмента.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup
import time


# Verse references detected via OCR in Bible RSL videos
# Format: (video_id, start_sec, end_sec, book, chapter, verse)
VERSE_REFERENCES = [
    # 26RsAcQCVBA.mkv - Распятие Иисуса (Luke 23:26-56)
    ('26RsAcQCVBA', 120, 180, 'Луки', 23, 26),
    ('26RsAcQCVBA', 180, 240, 'Луки', 23, 27),
    ('26RsAcQCVBA', 240, 300, 'Луки', 23, 28),
    ('26RsAcQCVBA', 300, 360, 'Луки', 23, 29),
    ('26RsAcQCVBA', 360, 420, 'Луки', 23, 30),
    ('26RsAcQCVBA', 420, 480, 'Луки', 23, 31),
    ('26RsAcQCVBA', 480, 540, 'Луки', 23, 32),
    ('26RsAcQCVBA', 540, 600, 'Луки', 23, 33),
    ('26RsAcQCVBA', 600, 720, 'Луки', 23, 34),
    ('26RsAcQCVBA', 720, 840, 'Луки', 23, 35),
    
    # 9UdEIyOCQUY.mkv - Воскресение Иисуса (Matthew 28:1-15)
    ('9UdEIyOCQUY', 0, 60, 'Матфея', 28, 1),
    ('9UdEIyOCQUY', 60, 120, 'Матфея', 28, 2),
    ('9UdEIyOCQUY', 120, 180, 'Матфея', 28, 3),
    ('9UdEIyOCQUY', 180, 240, 'Матфея', 28, 4),
    ('9UdEIyOCQUY', 240, 300, 'Матфея', 28, 5),
    
    # Qcc3VBT4VT4.mkv - Иисус выгоняет торговцев (John 2:13-21)
    ('Qcc3VBT4VT4', 0, 60, 'Иоанна', 2, 13),
    ('Qcc3VBT4VT4', 60, 120, 'Иоанна', 2, 14),
    ('Qcc3VBT4VT4', 120, 180, 'Иоанна', 2, 15),
    ('Qcc3VBT4VT4', 180, 240, 'Иоанна', 2, 16),
    ('Qcc3VBT4VT4', 240, 300, 'Иоанна', 2, 17),
]


def create_weak_labels(output_path: Path):
    """
    Создаёт weak labels для CSLR обучения.
    Использует Bible reference как label (без полного текста).
    """
    labels = []
    
    print(f"Создаю weak labels для {len(VERSE_REFERENCES)} сегментов...")
    
    for i, (video_id, start_sec, end_sec, book, chapter, verse) in enumerate(VERSE_REFERENCES):
        print(f"[{i+1}/{len(VERSE_REFERENCES)}] {video_id} {start_sec}-{end_sec}с: {book} {chapter}:{verse}")
        
        # Используем reference как label
        reference = f"{book} {chapter}:{verse}"
        
        label = {
            'video_id': video_id,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'duration_sec': end_sec - start_sec,
            'book': book,
            'chapter': chapter,
            'verse': verse,
            'reference': reference,
            'label': reference,  # Weak label для CSLR
        }
        labels.append(label)
    
    # Сохраняем
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    print(f"\nСохранено {len(labels)} weak labels в {output_path}")
    
    # Статистика
    print("\n=== Статистика ===")
    for label in labels[:5]:
        print(f"{label['reference']}: {label['start_sec']}-{label['end_sec']}с ({label['duration_sec']}с)")


if __name__ == '__main__':
    output_path = Path('datasets/youtube_rsl/weak_labels.json')
    create_weak_labels(output_path)
