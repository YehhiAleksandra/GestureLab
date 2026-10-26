#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Предсказание слов/фраз на видео с использованием полноценной модели.
Модель распознаёт и буквы дактиля, и слова РЖЯ.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import GRU, normalize


def load_model(model_path: str):
    """Загрузить обученную модель."""
    data = np.load(model_path, allow_pickle=True)
    names = data["names"]
    idx_to_label = {i: name for i, name in enumerate(names)}
    hidden_dim = int(data["hidden"])
    model = GRU(d=126, h=hidden_dim, c=len(names), seed=0)
    for key in ["Wz", "bz", "Wr", "br", "Wh", "bh", "Wo", "bo"]:
        model.p[key] = data[key]
    return model, idx_to_label


def predict_windows(model, idx_to_label, windows):
    """Предсказать метки для окон."""
    windows_norm = normalize(windows)
    prob, _ = model.forward(windows_norm)
    pred_idx = np.argmax(prob, axis=1)
    pred_labels = [idx_to_label[i] for i in pred_idx]
    confidences = np.max(prob, axis=1)
    return pred_labels, confidences


def main():
    parser = argparse.ArgumentParser(description="Предсказание слов на видео")
    parser.add_argument("landmarks_json", help="Путь к JSON с ландмарками")
    parser.add_argument("--model", default="translator_model_full.npz", help="Путь к модели")
    parser.add_argument("--output", default="predicted_text.json", help="Выходной JSON")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Минимальная уверенность")
    args = parser.parse_args()
    
    # Загружаем модель
    print(f"Загрузка модели: {args.model}")
    model, idx_to_label = load_model(args.model)
    print(f"Классов: {len(idx_to_label)}")
    
    # Загружаем ландмарки
    print(f"Загрузка ландмарок: {args.landmarks_json}")
    with open(args.landmarks_json, encoding="utf-8") as f:
        data = json.load(f)
    
    frames = data.get("frames", data) if isinstance(data, dict) else data
    print(f"Кадров: {len(frames)}")
    
    # Извлекаем окна (22 кадра)
    windows = []
    window_size = 22
    for i in range(len(frames) - window_size + 1):
        window_frames = frames[i:i + window_size]
        landmarks = []
        for frame in window_frames:
            lm = frame.get("landmarks", [])
            if len(lm) == 42:  # 21 точка × 2 руки
                flat = [coord for point in lm for coord in (point["x"], point["y"], point["z"])]
                landmarks.append(flat)
            elif len(lm) == 21:  # Только одна рука
                flat = [coord for point in lm for coord in (point["x"], point["y"], point["z"])]
                landmarks.append(flat + flat)  # Дублируем для второй руки
        if len(landmarks) == window_size:
            windows.append(landmarks)
    
    if not windows:
        print("Не найдено окон с жестами", file=sys.stderr)
        return
    
    windows = np.array(windows)
    print(f"Извлечено окон: {len(windows)}")
    
    # Предсказываем
    print("Предсказание...")
    pred_labels, confidences = predict_windows(model, idx_to_label, windows)
    
    # Фильтруем по уверенности
    filtered = [(label, conf) for label, conf in zip(pred_labels, confidences) if conf >= args.min_confidence]
    
    # Убираем повторы подряд
    result = []
    prev_label = None
    for label, conf in filtered:
        if label != prev_label and label != "no_event":
            result.append({"label": label, "confidence": float(conf)})
            prev_label = label
    
    # Сохраняем
    output = {
        "model": args.model,
        "total_windows": len(windows),
        "filtered_windows": len(result),
        "predictions": result,
        "text": " ".join([r["label"] for r in result])
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nПредсказано: {len(result)} слов/букв")
    print(f"Текст: {output['text']}")
    print(f"Сохранено: {args.output}")
    
    # Статистика
    letters = [r for r in result if len(r["label"]) == 1]
    words = [r for r in result if len(r["label"]) > 1]
    print(f"\nСтатистика:")
    print(f"  Букв: {len(letters)}")
    print(f"  Слов: {len(words)}")
    if words:
        print(f"  Примеры слов: {', '.join([w['label'] for w in words[:10]])}")


if __name__ == "__main__":
    main()
