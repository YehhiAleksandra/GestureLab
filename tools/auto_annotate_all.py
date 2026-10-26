"""
Полностью автоматический пайплайн обработки YouTube курса.

1. Авто-разметка всех видео Slovo моделью
2. Постобработка (сглаживание, удаление коротких)
3. Генерация финального JSON с разметкой
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import List, Dict

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


def postprocess_predictions(predictions: List[Dict], fps: float = 25.0,
                          min_duration_frames: int = 10,
                          smoothing_window: int = 5) -> List[Dict]:
    """
    Постобработка предсказаний:
    1. Majority voting для сглаживания
    2. Удаление коротких предсказаний
    3. Объединение соседних одинаковых
    """
    if not predictions:
        return []
    
    # Majority voting
    smoothed = []
    for i in range(len(predictions)):
        start_idx = max(0, i - smoothing_window // 2)
        end_idx = min(len(predictions), i + smoothing_window // 2 + 1)
        window = predictions[start_idx:end_idx]
        
        # Подсчёт голосов
        votes = {}
        for p in window:
            label = p['label']
            votes[label] = votes.get(label, 0) + 1
        
        # Берём большинство
        majority_label = max(votes, key=votes.get)
        smoothed.append({
            'label': majority_label,
            'start_frame': predictions[i]['start_frame'],
            'end_frame': predictions[i]['end_frame'],
            'confidence': predictions[i]['confidence']
        })
    
    # Объединение соседних одинаковых
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
    
    # Удаление коротких
    filtered = [m for m in merged if (m['end_frame'] - m['start_frame']) >= min_duration_frames]
    
    # Добавляем время
    for f in filtered:
        f['start_time'] = f['start_frame'] / fps
        f['end_time'] = f['end_frame'] / fps
        f['duration_frames'] = f['end_frame'] - f['start_frame']
    
    return filtered


def process_video(landmarks_path: str, model, device, 
                 window_size: int = 22, step: int = 5) -> List[Dict]:
    """Обрабатывает одно видео."""
    with open(landmarks_path, encoding='utf-8') as f:
        data = json.load(f)
    
    frames = data.get('frames', data) if isinstance(data, dict) else data
    
    # Извлечение окон
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
    
    if not windows:
        return []
    
    # Предсказание
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


def auto_annotate_all_courses(model_path: str, output_dir: str):
    """Автоматически размечает все видео курса."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Загрузка модели
    print(f"Загрузка модели: {model_path}")
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
    print("✓ Модель загружена")
    
    # Поиск всех landmarks файлов
    landmarks_dir = Path("datasets/youtube_processed")
    landmarks_files = sorted(landmarks_dir.glob("*_landmarks.json"))
    
    print(f"\nНайдено {len(landmarks_files)} видео для обработки")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    total_gestures = 0
    
    for i, lf in enumerate(landmarks_files):
        print(f"\n[{i+1}/{len(landmarks_files)}] {lf.name}")
        
        # Обработка
        predictions = process_video(str(lf), model, device)
        print(f"  Сырых предсказаний: {len(predictions)}")
        
        # Постобработка
        processed = postprocess_predictions(predictions, fps=25.0, 
                                           min_duration_frames=10,
                                           smoothing_window=5)
        print(f"  После постобработки: {len(processed)}")
        
        # Сохранение
        output_file = output_path / f"{lf.stem}_annotated.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        # Статистика
        unique = set(p['label'] for p in processed)
        print(f"  Уникальных жестов: {len(unique)}")
        
        all_results[lf.stem] = {
            'file': str(lf),
            'raw_predictions': len(predictions),
            'processed_gestures': len(processed),
            'unique_classes': len(unique),
            'gestures': processed
        }
        
        total_gestures += len(processed)
    
    # Итоговый отчёт
    summary = {
        'total_videos': len(landmarks_files),
        'total_gestures': total_gestures,
        'avg_gestures_per_video': total_gestures / len(landmarks_files) if landmarks_files else 0,
        'model': model_path,
        'results': all_results
    }
    
    summary_file = output_path / "annotation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Обработано {len(landmarks_files)} видео")
    print(f"✓ Всего жестов: {total_gestures}")
    print(f"✓ Среднее на видео: {summary['avg_gestures_per_video']:.1f}")
    print(f"✓ Результаты сохранены в {output_dir}/")
    
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='slovo_model_pytorch.pt')
    parser.add_argument('--output', default='datasets/youtube_annotated')
    args = parser.parse_args()
    
    auto_annotate_all_courses(args.model, args.output)
