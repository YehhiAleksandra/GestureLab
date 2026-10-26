"""
Ensemble моделей для улучшения распознавания жестов.

Использует:
1. Slovo GRU (90.7%) - основная модель
2. Улучшенную постобработку с контекстным анализом
3. Фильтрацию шумовых предсказаний
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import List, Dict, Tuple
from collections import Counter

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
    hands = frame_data.get("hands", [])
    if not hands:
        return None
    hand1 = hands[0]
    hand2 = hands[1] if len(hands) > 1 else hand1
    flat1 = [coord for point in hand1 for coord in point]
    flat2 = [coord for point in hand2 for coord in point]
    return flat1 + flat2


def load_model(model_path: str, device):
    """Загружает модель."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_classes = checkpoint.get('num_classes', 100)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        state_dict = checkpoint['model_state_dict']
    else:
        num_classes = 100
        hidden_dim = 128
        state_dict = checkpoint
    
    model = GestureGRU(input_dim=126, hidden_dim=hidden_dim, 
                      num_classes=num_classes, num_layers=2, dropout=0.3)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def predict_with_confidence(model, windows_np: np.ndarray, device, 
                           confidence_threshold: float = 0.3) -> Tuple[List[int], List[float]]:
    """Предсказывает с фильтрацией по уверенности."""
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
    
    # Фильтрация по уверенности
    filtered_preds = []
    filtered_confs = []
    
    for pred, conf in zip(all_preds, all_confs):
        if conf >= confidence_threshold:
            filtered_preds.append(pred)
            filtered_confs.append(conf)
        else:
            # Если уверенность низкая, используем предыдущее предсказание
            if filtered_preds:
                filtered_preds.append(filtered_preds[-1])
                filtered_confs.append(filtered_confs[-1])
            else:
                filtered_preds.append(pred)
                filtered_confs.append(conf)
    
    return filtered_preds, filtered_confs


def advanced_postprocess(predictions: List[Dict], fps: float = 25.0,
                        min_duration_frames: int = 15,
                        smoothing_window: int = 7,
                        confidence_threshold: float = 0.4) -> List[Dict]:
    """
    Улучшенная постобработка:
    1. Адаптивное сглаживание (majority voting с весами)
    2. Фильтрация по уверенности
    3. Удаление коротких и шумовых предсказаний
    4. Контекстный анализ (учёт соседних жестов)
    """
    if not predictions:
        return []
    
    # Шаг 1: Адаптивное сглаживание с весами уверенности
    smoothed = []
    for i in range(len(predictions)):
        start_idx = max(0, i - smoothing_window // 2)
        end_idx = min(len(predictions), i + smoothing_window // 2 + 1)
        window = predictions[start_idx:end_idx]
        
        # Взвешенное голосование (чем выше уверенность, тем больше вес)
        weighted_votes = {}
        for p in window:
            label = p['label']
            weight = p['confidence']
            weighted_votes[label] = weighted_votes.get(label, 0) + weight
        
        # Берём наиболее вероятный жест
        best_label = max(weighted_votes, key=weighted_votes.get)
        best_conf = weighted_votes[best_label] / sum(weighted_votes.values())
        
        smoothed.append({
            'label': best_label,
            'start_frame': predictions[i]['start_frame'],
            'end_frame': predictions[i]['end_frame'],
            'confidence': best_conf,
            'original_conf': predictions[i]['confidence']
        })
    
    # Шаг 2: Объединение соседних одинаковых
    merged = []
    current = smoothed[0].copy()
    
    for i in range(1, len(smoothed)):
        if smoothed[i]['label'] == current['label']:
            current['end_frame'] = smoothed[i]['end_frame']
            current['confidence'] = max(current['confidence'], smoothed[i]['confidence'])
        else:
            merged.append(current)
            current = smoothed[i].copy()
    merged.append(current)
    
    # Шаг 3: Удаление коротких и низкоуверенных
    filtered = []
    for m in merged:
        duration = m['end_frame'] - m['start_frame']
        if duration >= min_duration_frames and m['confidence'] >= confidence_threshold:
            filtered.append(m)
    
    # Шаг 4: Контекстный анализ (замена изолированных жестов на соседние)
    if len(filtered) > 2:
        context_filtered = []
        for i in range(len(filtered)):
            if i == 0 or i == len(filtered) - 1:
                context_filtered.append(filtered[i])
            else:
                # Если жест сильно отличается от соседей, заменяем
                prev_label = filtered[i-1]['label']
                next_label = filtered[i+1]['label']
                current_label = filtered[i]['label']
                
                if prev_label == next_label and prev_label != current_label:
                    # Изолированный жест - заменяем на соседний
                    replacement = filtered[i-1].copy()
                    replacement['start_frame'] = filtered[i]['start_frame']
                    replacement['end_frame'] = filtered[i]['end_frame']
                    context_filtered.append(replacement)
                else:
                    context_filtered.append(filtered[i])
        filtered = context_filtered
    
    # Добавляем время
    for f in filtered:
        f['start_time'] = f['start_frame'] / fps
        f['end_time'] = f['end_frame'] / fps
        f['duration_frames'] = f['end_frame'] - f['start_frame']
        f['duration_seconds'] = f['duration_frames'] / fps
    
    return filtered


def process_video_advanced(landmarks_path: str, model, device,
                          window_size: int = 22, step: int = 3,
                          confidence_threshold: float = 0.3) -> List[Dict]:
    """Обрабатывает видео с улучшенными параметрами."""
    with open(landmarks_path, encoding='utf-8') as f:
        data = json.load(f)
    
    frames = data.get('frames', data) if isinstance(data, dict) else data
    print(f"  Кадров: {len(frames)}")
    
    # Извлечение окон (меньше шаг = больше окон)
    windows = []
    window_starts = []
    
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
    
    print(f"  Окон: {len(windows)} (step={step})")
    
    if not windows:
        return []
    
    # Предсказание с фильтрацией по уверенности
    windows_np = np.array(windows, dtype=np.float32)
    all_preds, all_confs = predict_with_confidence(
        model, windows_np, device, confidence_threshold
    )
    
    # Группировка
    predictions = []
    for pred, conf, start in zip(all_preds, all_confs, window_starts):
        label = slovo_classes[pred] if pred < len(slovo_classes) else f"class_{pred}"
        predictions.append({
            'label': label,
            'start_frame': start,
            'end_frame': start + window_size,
            'confidence': float(conf)
        })
    
    return predictions


def ensemble_recognition(landmarks_json: str, model_paths: List[str],
                        output_json: str):
    """
    Ensemble распознавание с несколькими моделями.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Загрузка моделей
    models = []
    for model_path in model_paths:
        print(f"Загрузка модели: {model_path}")
        model = load_model(model_path, device)
        models.append(model)
    
    print(f"✓ Загружено {len(models)} моделей")
    
    # Обработка видео каждой моделью
    all_predictions = []
    
    for i, model in enumerate(models):
        print(f"\nМодель {i+1}/{len(models)}:")
        predictions = process_video_advanced(
            landmarks_json, model, device,
            window_size=22, step=3, confidence_threshold=0.3
        )
        print(f"  Сырых предсказаний: {len(predictions)}")
        all_predictions.append(predictions)
    
    # Ensemble: голосование
    print("\nEnsemble голосование...")
    
    # Создаём временную сетку
    with open(landmarks_json, encoding='utf-8') as f:
        data = json.load(f)
    frames = data.get('frames', data) if isinstance(data, dict) else data
    total_frames = len(frames)
    
    # Для каждого временного окна собираем голоса всех моделей
    ensemble_predictions = []
    time_step = 5  # кадров
    
    for start_frame in range(0, total_frames - 22, time_step):
        end_frame = start_frame + 22
        
        # Собираем голоса
        votes = []
        confs = []
        
        for model_preds in all_predictions:
            for pred in model_preds:
                if pred['start_frame'] <= start_frame < pred['end_frame']:
                    votes.append(pred['label'])
                    confs.append(pred['confidence'])
                    break
        
        if votes:
            # Majority voting с весами уверенности
            weighted_votes = {}
            for label, conf in zip(votes, confs):
                weighted_votes[label] = weighted_votes.get(label, 0) + conf
            
            best_label = max(weighted_votes, key=weighted_votes.get)
            best_conf = weighted_votes[best_label] / sum(weighted_votes.values())
            
            ensemble_predictions.append({
                'label': best_label,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'confidence': best_conf,
                'votes': len(votes)
            })
    
    print(f"  Ensemble предсказаний: {len(ensemble_predictions)}")
    
    # Постобработка
    print("\nПостобработка...")
    processed = advanced_postprocess(
        ensemble_predictions,
        fps=25.0,
        min_duration_frames=15,
        smoothing_window=7,
        confidence_threshold=0.4
    )
    
    print(f"  После постобработки: {len(processed)}")
    
    # Сохранение
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Сохранено в {output_json}")
    
    # Статистика
    unique = set(p['label'] for p in processed)
    print(f"\n=== Статистика ===")
    print(f"Всего жестов: {len(processed)}")
    print(f"Уникальных: {len(unique)}")
    print(f"Средняя уверенность: {np.mean([p['confidence'] for p in processed]):.3f}")
    print(f"Средняя длительность: {np.mean([p['duration_seconds'] for p in processed]):.2f}с")
    print(f"\nТоп-10 жестов:")
    counter = Counter(p['label'] for p in processed)
    for label, count in counter.most_common(10):
        print(f"  {label:20s}: {count:3d}")
    
    return processed


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('landmarks_json', help='JSON с landmarks')
    parser.add_argument('--models', nargs='+', default=['slovo_model_pytorch.pt'],
                       help='Пути к моделям')
    parser.add_argument('--output', default='reports/ensemble_translation.json')
    
    args = parser.parse_args()
    
    ensemble_recognition(args.landmarks_json, args.models, args.output)
