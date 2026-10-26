"""
Генерация перевода с использованием Slovo модели (90.7% accuracy).
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'slovo_repo'))
from constants import classes as slovo_classes


class GestureGRU(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=128, num_classes=100, num_layers=2, dropout=0.3):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                         num_layers=num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        B, T, C = x.shape
        x = x.reshape(B * T, C)
        x = self.batch_norm(x)
        x = x.reshape(B, T, C)
        output, _ = self.gru(x)
        last = output[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


def extract_landmarks(frame_data):
    """Извлекает landmarks из кадра."""
    hands = frame_data.get("hands", [])
    if not hands:
        return None
    hand1 = hands[0]
    hand2 = hands[1] if len(hands) > 1 else hand1
    flat1 = [coord for point in hand1 for coord in point]
    flat2 = [coord for point in hand2 for coord in point]
    return flat1 + flat2


def generate_translation(landmarks_json: str, model_path: str, output_json: str):
    """Генерирует перевод видео."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Загрузка модели
    print(f"Загрузка модели: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Проверяем формат сохранения
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_classes = checkpoint.get('num_classes', 100)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        state_dict = checkpoint['model_state_dict']
        print(f"  Checkpoint: num_classes={num_classes}, hidden_dim={hidden_dim}")
    else:
        num_classes = 100
        hidden_dim = 128
        state_dict = checkpoint
    
    model = GestureGRU(input_dim=126, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=2, dropout=0.3)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✓ Модель загружена")
    
    # Загрузка landmarks
    print(f"Загрузка landmarks: {landmarks_json}")
    with open(landmarks_json, encoding='utf-8') as f:
        data = json.load(f)
    
    frames = data.get('frames', data) if isinstance(data, dict) else data
    print(f"Кадров: {len(frames)}")
    
    # Извлечение окон
    print("Извлечение окон...")
    windows = []
    window_starts = []
    window_size = 22
    step = 5
    
    for i in range(0, len(frames) - window_size + 1, step):
        window_frames = frames[i:i + window_size]
        landmarks = []
        for frame in window_frames:
            lm = extract_landmarks(frame)
            if lm and len(lm) == 126:
                landmarks.append(lm)
        if len(landmarks) == window_size:
            windows.append(landmarks)
            window_starts.append(i)
    
    print(f"✓ Извлечено {len(windows)} окон")
    
    if not windows:
        print("❌ Нет окон для предсказания")
        return
    
    # Предсказание
    print("\nПредсказание...")
    windows_np = np.array(windows, dtype=np.float32)
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
    
    # Группировка предсказаний
    print("Группировка...")
    gestures = []
    current_label = None
    current_start = None
    current_conf = 0.0
    current_count = 0
    
    for i, (pred, conf, start) in enumerate(zip(all_preds, all_confs, window_starts)):
        label = slovo_classes[pred] if pred < len(slovo_classes) else f"class_{pred}"
        
        if label != current_label:
            if current_label is not None and current_count >= 2:
                end = window_starts[i - 1] + window_size
                avg_conf = current_conf / current_count
                gestures.append({
                    'label': current_label,
                    'start_frame': current_start,
                    'end_frame': end,
                    'confidence': f"{avg_conf:.3f}",
                    'duration_frames': end - current_start
                })
            current_label = label
            current_start = start
            current_conf = conf
            current_count = 1
        else:
            current_conf += conf
            current_count += 1
    
    # Последний жест
    if current_label is not None and current_count >= 2:
        end = window_starts[-1] + window_size
        avg_conf = current_conf / current_count
        gestures.append({
            'label': current_label,
            'start_frame': current_start,
            'end_frame': end,
            'confidence': f"{avg_conf:.3f}",
            'duration_frames': end - current_start
        })
    
    # Фильтрация коротких
    gestures = [g for g in gestures if g['duration_frames'] >= 10]
    
    # Сохранение
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(gestures, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Сохранено {len(gestures)} жестов в {output_json}")
    
    # Вывод
    print("\n=== Перевод ===")
    text = ' '.join([g['label'] for g in gestures])
    print(text)
    
    unique = set(g['label'] for g in gestures)
    print(f"\nСтатистика:")
    print(f"  Всего жестов: {len(gestures)}")
    print(f"  Уникальных: {len(unique)}")
    print(f"  Примеры: {list(unique)[:15]}")
    
    return gestures


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('landmarks_json')
    parser.add_argument('--model', default='slovo_model_pytorch.pt')
    parser.add_argument('--output', default='reports/slovo_translation.json')
    args = parser.parse_args()
    
    generate_translation(args.landmarks_json, args.model, args.output)
