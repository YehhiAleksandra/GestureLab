"""
Создать видео-сегмент со всеми уникальными жестами (без подписей).
Каждый жест показан один раз.
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def extract_best_segments(gestures_json, min_duration=1.5, max_duration=3.0):
    """Извлечь лучший сегмент для каждого уникального жеста."""
    with open(gestures_json, encoding='utf-8') as f:
        gestures = json.load(f)
    
    # Группируем по label
    by_label = defaultdict(list)
    for g in gestures:
        by_label[g['label']].append(g)
    
    # Для каждого класса берем сегмент близкий к max_duration
    segments = []
    for label, items in by_label.items():
        # Фильтруем по минимальной длительности
        valid = [g for g in items if g['duration_seconds'] >= min_duration]
        if not valid:
            valid = items
        
        # Ищем сегмент близкий к max_duration
        best = min(valid, key=lambda g: abs(g['duration_seconds'] - max_duration))
        
        # Ограничиваем длительность
        duration = min(best['duration_seconds'], max_duration)
        
        segments.append({
            'label': label,
            'start_time': best['start_time'],
            'duration': duration,
            'confidence': best['confidence']
        })
    
    # Сортируем по алфавиту
    segments.sort(key=lambda s: s['label'])
    return segments


def extract_segment(video_path, segment, output_path):
    """Извлечь сегмент через ffmpeg."""
    start = segment['start_time']
    duration = segment['duration']
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', video_path,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-r', '25',
        '-s', '640x480',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def concatenate_videos(video_files, output_path):
    """Склеить видео через ffmpeg concat."""
    if not video_files:
        return False
    
    # Создаем файл списка
    list_file = Path(output_path).parent / 'concat_list.txt'
    with open(list_file, 'w', encoding='utf-8') as f:
        for vf in video_files:
            abs_path = Path(vf).absolute()
            f.write(f"file '{abs_path}'\n")
    
    # Склеиваем
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_file),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink()
    
    return result.returncode == 0


def main():
    video_path = 'reports/holy_sepulchre_final_video.mp4'
    gestures_json = 'reports/holy_sepulchre_final.json'
    output_dir = Path('reports/segments')
    output_dir.mkdir(exist_ok=True)
    
    print("Извлечение лучших сегментов...")
    segments = extract_best_segments(gestures_json, min_duration=1.5, max_duration=3.0)
    print(f"Найдено {len(segments)} уникальных жестов")
    
    # Создаем сегменты
    segment_files = []
    for i, seg in enumerate(segments):
        print(f"[{i+1}/{len(segments)}] {seg['label']} ({seg['duration']:.1f}s, conf={seg['confidence']:.0%})...")
        
        output_path = output_dir / f"segment_{i:03d}.mp4"
        if extract_segment(video_path, seg, output_path):
            segment_files.append(str(output_path))
        else:
            print(f"  Ошибка создания сегмента", file=sys.stderr)
    
    if not segment_files:
        print("Не создано ни одного сегмента", file=sys.stderr)
        return
    
    # Склеиваем
    print(f"\nСклейка {len(segment_files)} сегментов...")
    final_output = 'reports/all_gestures_showcase.mp4'
    if concatenate_videos(segment_files, final_output):
        print(f"\n✓ Готово: {final_output}")
        
        # Информация
        total_duration = sum(s['duration'] for s in segments)
        print(f"  Уникальных жестов: {len(segments)}")
        print(f"  Общая длительность: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        
        # Список жестов
        print(f"\n  Жесты:")
        for s in segments:
            print(f"    • {s['label']} ({s['duration']:.1f}s, {s['confidence']:.0%})")
        
        # Сохраняем список жестов в JSON
        list_json = Path(final_output).with_suffix('.json')
        with open(list_json, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"\n  Список жестов: {list_json}")
    else:
        print("Ошибка склейки видео", file=sys.stderr)


if __name__ == '__main__':
    main()
