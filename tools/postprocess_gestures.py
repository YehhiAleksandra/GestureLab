"""
Улучшенная постобработка распознавания жестов.
Фильтрует шум, убирает повторы, строит связную последовательность.
"""
import json
import sys
from pathlib import Path


def filter_gestures(gestures, min_confidence=0.80, min_duration=2.0, max_duration=8.0, min_gap=5.0):
    """
    Фильтрация жестов:
    - min_confidence: минимальная уверенность
    - min_duration: минимальная длительность (сек)
    - max_duration: максимальная длительность (сек) — фильтрует длинные паузы
    - min_gap: минимальный промежуток между одинаковыми жестами (сек)
    """
    filtered = []
    last_seen = {}  # label -> last_end_time
    
    for g in gestures:
        label = g['label']
        conf = g['confidence']
        duration = g['duration_seconds']
        start = g['start_time']
        end = g['end_time']
        
        # Фильтр по уверенности
        if conf < min_confidence:
            continue
        
        # Фильтр по длительности
        if duration < min_duration:
            continue
        
        # Фильтр слишком длинных (это не жесты, а паузы/фон)
        if duration > max_duration:
            continue
        
        # Фильтр повторов
        if label in last_seen:
            gap = start - last_seen[label]
            if gap < min_gap:
                continue
        
        filtered.append(g)
        last_seen[label] = end
    
    return filtered


def build_text_sequence(gestures):
    """Построить текстовую последовательность из жестов."""
    # Разделяем на буквы и слова
    letters = []
    words = []
    
    for g in gestures:
        label = g['label']
        if len(label) == 1 and label.isalpha():
            letters.append(g)
        else:
            words.append(g)
    
    return letters, words


def main():
    input_json = 'reports/holy_sepulchre_final.json'
    output_json = 'reports/holy_sepulchre_filtered.json'
    
    with open(input_json, encoding='utf-8') as f:
        gestures = json.load(f)
    
    print(f"Исходных жестов: {len(gestures)}")
    
    # Фильтрация
    filtered = filter_gestures(
        gestures,
        min_confidence=0.80,
        min_duration=2.0,
        max_duration=8.0,
        min_gap=5.0
    )
    
    print(f"После фильтрации: {len(filtered)}")
    print(f"Отфильтровано: {len(gestures) - len(filtered)} ({100*(len(gestures)-len(filtered))/len(gestures):.1f}%)")
    print()
    
    # Разделяем на буквы и слова
    letters, words = build_text_sequence(filtered)
    
    print(f"Букв: {len(letters)}")
    print(f"Слов: {len(words)}")
    print()
    
    # Показываем последовательность
    print("Последовательность жестов (отфильтрованная):")
    for i, g in enumerate(filtered):
        marker = "📝" if len(g['label']) > 1 else "🔤"
        print(f"{i+1:3d}. [{g['start_time']:6.1f}s-{g['end_time']:6.1f}s] {marker} {g['label']:15s} (conf={g['confidence']:.0%}, dur={g['duration_seconds']:.1f}s)")
    
    # Текстовая последовательность
    print("\n" + "="*60)
    print("Текстовая последовательность:")
    print("="*60)
    
    # Строим текст
    text_parts = []
    for g in filtered:
        text_parts.append(g['label'])
    
    text = " ".join(text_parts)
    print(f"\n{text}")
    print(f"\nВсего элементов: {len(text_parts)}")
    
    # Сохраняем
    output = {
        'original_count': len(gestures),
        'filtered_count': len(filtered),
        'letters_count': len(letters),
        'words_count': len(words),
        'text': text,
        'gestures': filtered
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Сохранено: {output_json}")


if __name__ == '__main__':
    main()
