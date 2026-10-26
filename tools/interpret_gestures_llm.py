"""
Использовать LLM для построения связного текста из распознанных жестов.
"""
import json
import sys
from pathlib import Path

# Добавляем путь для импорта из tools
sys.path.insert(0, str(Path(__file__).parent))


def interpret_with_llm(gestures_text, video_context=""):
    """
    Использовать LLM для интерпретации жестов.
    Пока заглушка — нужно подключить реальный LLM API.
    """
    prompt = f"""
Ты — эксперт по русскому жестовому языку (РЖЯ).

Видео: {video_context}

Распознанные жесты (последовательность):
{gestures_text}

Задача:
1. Проанализируй последовательность жестов
2. Построй связный текст, который мог бы быть показан на видео
3. Учти контекст видео и тему
4. Если жесты не соответствуют теме, предложи альтернативную интерпретацию

Верни JSON:
{{
  "interpretation": "связный текст",
  "confidence": 0.0-1.0,
  "notes": "комментарии к интерпретации"
}}
"""
    print(prompt)
    return {
        "interpretation": "Требуется подключение LLM API",
        "confidence": 0.0,
        "notes": "Скрипт заглушка — нужна интеграция с OpenRouter/HuggingFace"
    }


def main():
    input_json = 'reports/holy_sepulchre_filtered.json'
    
    with open(input_json, encoding='utf-8') as f:
        data = json.load(f)
    
    gestures = data['gestures']
    text = data['text']
    
    print(f"Распознанных жестов: {len(gestures)}")
    print(f"Текст: {text}")
    print()
    
    # Контекст видео
    video_context = "Видео 'holy_sepulchre' — вероятно, про Святой Гроб (Храм Гроба Господня в Иерусалиме). Тема: религиозная, историческая."
    
    # Интерпретация через LLM
    result = interpret_with_llm(text, video_context)
    
    print("\n" + "="*60)
    print("Интерпретация:")
    print("="*60)
    print(f"Текст: {result['interpretation']}")
    print(f"Уверенность: {result['confidence']:.0%}")
    print(f"Примечания: {result['notes']}")
    
    # Сохраняем
    output_json = Path(input_json).with_name('holy_sepulchre_interpreted.json')
    output = {
        **data,
        'interpretation': result
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Сохранено: {output_json}")


if __name__ == '__main__':
    main()
