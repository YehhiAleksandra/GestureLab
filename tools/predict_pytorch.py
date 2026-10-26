#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Предсказание слов на видео с PyTorch моделью.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


class GestureGRU(nn.Module):
    """GRU модель (та же архитектура что при обучении)."""
    
    def __init__(self, input_dim=126, hidden_dim=128, num_classes=100, num_layers=2, dropout=0.3):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = self.batch_norm(x)
        x = x.reshape(batch_size, seq_len, -1)
        output, hidden = self.gru(x)
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output


def load_model(model_path: str, device):
    """Загрузить PyTorch модель."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = checkpoint['num_classes']
    hidden_dim = checkpoint['hidden_dim']
    
    model = GestureGRU(input_dim=126, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Загружаем имена классов из датасета
    data = np.load('datasets/combined_top100.npz', allow_pickle=True)
    names = data['names']
    idx_to_label = {i: str(name) for i, name in enumerate(names)}
    
    return model, idx_to_label


def extract_landmarks_from_frame(frame_data):
    """Извлечь ландмарки из кадра."""
    hands = frame_data.get("hands", [])
    if not hands:
        return None
    hand1 = hands[0]
    hand2 = hands[1] if len(hands) > 1 else hand1
    flat1 = [coord for point in hand1 for coord in point]
    flat2 = [coord for point in hand2 for coord in point]
    return flat1 + flat2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("landmarks_json")
    parser.add_argument("--model", default="translator_model_pytorch.pt")
    parser.add_argument("--output", default="predicted_pytorch.json")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"Загрузка модели: {args.model}")
    model, idx_to_label = load_model(args.model, device)
    print(f"Классов: {len(idx_to_label)}")
    
    print(f"Загрузка ландмарок: {args.landmarks_json}")
    with open(args.landmarks_json, encoding="utf-8") as f:
        data = json.load(f)
    
    frames = data.get("frames", data)
    print(f"Кадров: {len(frames)}")
    
    # Извлекаем окна
    windows = []
    window_size = 22
    for i in range(len(frames) - window_size + 1):
        window_frames = frames[i:i + window_size]
        landmarks = []
        for frame in window_frames:
            lm = extract_landmarks_from_frame(frame)
            if lm and len(lm) == 126:
                landmarks.append(lm)
        if len(landmarks) == window_size:
            windows.append(landmarks)
    
    if not windows:
        print("Не найдено окон с жестами", file=sys.stderr)
        return
    
    windows_np = np.array(windows)
    print(f"Извлечено окон: {len(windows_np)}")
    
    # Предсказываем батчами
    print("Предсказание...")
    batch_size = 256
    all_preds = []
    all_confs = []
    
    with torch.no_grad():
        for i in range(0, len(windows_np), batch_size):
            batch = torch.FloatTensor(windows_np[i:i+batch_size]).to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
    
    # Фильтруем
    result = []
    prev_label = None
    for pred_idx, conf in zip(all_preds, all_confs):
        label = idx_to_label[pred_idx]
        if conf >= args.min_confidence and label != prev_label and label != "no_event":
            result.append({"label": label, "confidence": float(conf)})
            prev_label = label
    
    output = {
        "model": args.model,
        "total_windows": len(windows_np),
        "filtered_windows": len(result),
        "predictions": result,
        "text": " ".join([r["label"] for r in result])
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nПредсказано: {len(result)} слов/букв")
    print(f"Текст: {output['text'][:300]}")
    print(f"Сохранено: {args.output}")
    
    letters = [r for r in result if len(r["label"]) == 1]
    words = [r for r in result if len(r["label"]) > 1]
    print(f"\nСтатистика: Букв: {len(letters)}, Слов: {len(words)}")
    if words:
        unique_words = list(set([w['label'] for w in words]))
        print(f"Уникальных слов: {len(unique_words)}")
        print(f"Примеры: {unique_words[:20]}")


if __name__ == "__main__":
    main()
