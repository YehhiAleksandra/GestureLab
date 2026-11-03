# -*- coding: utf-8 -*-
"""Сегментация VOGinfo видео по VTT-таймингам для извлечения 89 слов.

Парсит VTT-субтитры, находит вхождения слов из списка, вырезает видео-сегменты
через ffmpeg. Сохраняет метаданные для дальнейшего использования в обучении.

Usage:
  python experiments/segment_voginfo.py --dry-run  # тест без вырезки
  python experiments/segment_voginfo.py            # полная обработка
"""
import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VOGINFO_SUBS = Path('D:/hfdata/ytsl25_rsl/subs')
VOGINFO_VIDEOS = Path('D:/hfdata/ytsl25_rsl/videos')
OUTPUT_DIR = Path('D:/hfdata/voginfo_segments')
COVERAGE_FILE = Path('D:/hfdata/voginfo_coverage.json')
METADATA_FILE = OUTPUT_DIR / 'segments_metadata.json'


def parse_vtt(vtt_path):
    """Парсит VTT-файл, возвращает список (start_sec, end_sec, text)."""
    text = vtt_path.read_text(encoding='utf-8')
    
    # Убираем заголовок WEBVTT
    text = re.sub(r'^WEBVTT[^\n]*\n', '', text)
    
    segments = []
    current_time = None
    current_text = []
    
    for line in text.split('\n'):
        line = line.strip()
        
        # Парсим тайминг
        time_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})',
            line
        )
        if time_match:
            # Сохраняем предыдущий сегмент
            if current_time and current_text:
                segments.append({
                    'start': current_time[0],
                    'end': current_time[1],
                    'text': ' '.join(current_text).lower()
                })
            
            # Новый тайминг
            h1, m1, s1, ms1, h2, m2, s2, ms2 = time_match.groups()
            start = int(h1)*3600 + int(m1)*60 + int(s1) + int(ms1)/1000
            end = int(h2)*3600 + int(m2)*60 + int(s2) + int(ms2)/1000
            current_time = (start, end)
            current_text = []
        elif line and not line.isdigit():
            # Убираем VTT-разметку
            line = re.sub(r'<[^>]+>', '', line)
            if line:
                current_text.append(line)
    
    # Последний сегмент
    if current_time and current_text:
        segments.append({
            'start': current_time[0],
            'end': current_time[1],
            'text': ' '.join(current_text).lower()
        })
    
    return segments


def find_word_occurrences(segments, target_words):
    """Находит вхождения слов из target_words в segments.
    
    Возвращает dict: word -> [(segment_idx, start, end, context)]
    """
    occurrences = defaultdict(list)
    
    for seg_idx, seg in enumerate(segments):
        text = seg['text']
        words_in_text = re.findall(r'[а-яё]+', text)
        
        for word in target_words:
            if word in words_in_text:
                occurrences[word].append({
                    'segment_idx': seg_idx,
                    'start': seg['start'],
                    'end': seg['end'],
                    'context': text
                })
    
    return occurrences


def extract_video_segment(video_path, start, end, output_path, padding=0.5):
    """Вырезает сегмент видео через ffmpeg с отступом padding секунды."""
    # Добавляем отступ
    start = max(0, start - padding)
    end = end + padding
    duration = end - start
    
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-ss', f'{start:.3f}',
        '-t', f'{duration:.3f}',
        '-c', 'copy',
        '-y',  # перезаписать
        str(output_path)
    ]
    
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=30
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def main():
    parser = argparse.ArgumentParser(description='Сегментация VOGinfo видео')
    parser.add_argument('--dry-run', action='store_true', help='Тест без вырезки')
    parser.add_argument('--max-videos', type=int, default=None, help='Максимум видео')
    args = parser.parse_args()
    
    # Загружаем список слов
    coverage = json.load(open(COVERAGE_FILE, encoding='utf-8'))
    target_words = set(coverage['found_list'])
    print(f'Целевых слов: {len(target_words)}')
    
    # Создаём выходную директорию
    OUTPUT_DIR.mkdir(exist_ok=True)
    for word in target_words:
        (OUTPUT_DIR / word).mkdir(exist_ok=True)
    
    # Собираем статистику
    all_occurrences = defaultdict(list)  # word -> [(video_id, start, end)]
    videos_processed = 0
    videos_with_words = 0
    
    vtt_files = sorted(VOGINFO_SUBS.glob('*.vtt'))
    if args.max_videos:
        vtt_files = vtt_files[:args.max_videos]
    
    print(f'Всего VTT-файлов: {len(vtt_files)}')
    
    for vtt_idx, vtt_path in enumerate(vtt_files):
        if (vtt_idx + 1) % 50 == 0:
            print(f'Обработано {vtt_idx + 1} / {len(vtt_files)} файлов')
        
        video_id = vtt_path.stem.replace('.ru', '')
        video_path = VOGINFO_VIDEOS / f'{video_id}.mp4'
        
        if not video_path.exists():
            continue
        
        videos_processed += 1
        
        # Парсим VTT
        try:
            segments = parse_vtt(vtt_path)
        except Exception as e:
            print(f'  Ошибка парсинга {vtt_path.name}: {e}')
            continue
        
        # Ищем слова
        occurrences = find_word_occurrences(segments, target_words)
        
        if not occurrences:
            continue
        
        videos_with_words += 1
        
        # Для каждого найденного слова
        for word, occs in occurrences.items():
            for occ in occs:
                all_occurrences[word].append({
                    'video_id': video_id,
                    'start': occ['start'],
                    'end': occ['end'],
                    'context': occ['context']
                })
                
                if not args.dry_run:
                    # Вырезаем сегмент
                    seg_idx = len([o for o in all_occurrences[word] if o['video_id'] == video_id])
                    output_name = f'{video_id}_{seg_idx:03d}.mp4'
                    output_path = OUTPUT_DIR / word / output_name
                    
                    if not output_path.exists():
                        success = extract_video_segment(
                            video_path,
                            occ['start'],
                            occ['end'],
                            output_path
                        )
                        if not success:
                            print(f'  Ошибка вырезки {output_name}')
    
    # Сохраняем метаданные
    metadata = {
        'created': '2026-11-03',
        'source': 'VOGinfo YouTube-SL-25',
        'target_words': len(target_words),
        'videos_processed': videos_processed,
        'videos_with_words': videos_with_words,
        'total_occurrences': sum(len(v) for v in all_occurrences.values()),
        'occurrences_by_word': {
            word: {
                'count': len(occs),
                'unique_videos': len(set(o['video_id'] for o in occs))
            }
            for word, occs in all_occurrences.items()
        },
        'occurrences': dict(all_occurrences)
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f'\n=== Итог ===')
    print(f'Видео обработано: {videos_processed}')
    print(f'Видео с целевыми словами: {videos_with_words}')
    print(f'Всего вхождений: {metadata["total_occurrences"]}')
    print(f'Слов найдено: {len(all_occurrences)}')
    print(f'Метаданные сохранены: {METADATA_FILE}')
    
    if args.dry_run:
        print('\n[DRY RUN] Видео не вырезаны. Запусти без --dry-run для извлечения.')
    
    # Топ-10 частых слов
    print(f'\nТоп-10 частых слов:')
    sorted_words = sorted(all_occurrences.items(), key=lambda x: len(x[1]), reverse=True)
    for word, occs in sorted_words[:10]:
        print(f'  {word}: {len(occs)} вхождений в {metadata["occurrences_by_word"][word]["unique_videos"]} видео')


if __name__ == '__main__':
    main()
