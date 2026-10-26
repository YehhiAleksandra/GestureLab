#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пошаговый переводчик: визуализация landmarks + синхронное предсказание.
Для каждого скользящего окна — предсказание + таймкод.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


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
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = self.batch_norm(x)
        x = x.reshape(batch_size, seq_len, -1)
        output, _ = self.gru(x)
        output = output[:, -1, :]
        output = self.dropout(output)
        return self.fc(output)


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = GestureGRU(input_dim=126, hidden_dim=checkpoint['hidden_dim'],
                       num_classes=checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    data = np.load('datasets/combined_top100.npz', allow_pickle=True)
    idx_to_label = {i: str(n) for i, n in enumerate(data['names'])}
    return model, idx_to_label


def draw_landmarks(frame, hands, color=(0, 255, 0)):
    """Рисуем точки и связи на руках."""
    # Связи между точками (21 точка на руку)
    connections = [
        (0,1),(1,2),(2,3),(3,4),       # большой палец
        (0,5),(5,6),(6,7),(7,8),       # указательный
        (0,9),(9,10),(10,11),(11,12),  # средний
        (0,13),(13,14),(14,15),(15,16),# безымянный
        (0,17),(17,18),(18,19),(19,20) # мизинец
    ]
    
    h, w = frame.shape[:2]
    for hand in hands:
        points = [(int(p[0]*w), int(p[1]*h)) for p in hand]
        # Рисуем связи
        for i, j in connections:
            cv2.line(frame, points[i], points[j], color, 2)
        # Рисуем точки
        for p in points:
            cv2.circle(frame, p, 4, color, -1)
    return frame


def process_video_step_by_step(video_path, model, idx_to_label, device, 
                                output_json, output_frames_dir,
                                window_size=22, step=5, min_conf=0.4):
    """Обрабатываем видео поэтапно: скользящее окно с шагом."""
    import mediapipe as mp
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        model_complexity=0, min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Видео: {video_path}")
    print(f"FPS: {fps}, Кадров: {total_frames}, Длительность: {total_frames/fps:.1f}с")
    
    # Собираем все кадры с landmarks
    all_frames_data = []
    frame_idx = 0
    print("Извлечение landmarks...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                hands_data.append(hand)
        
        all_frames_data.append({
            'frame_idx': frame_idx,
            'timestamp': frame_idx / fps,
            'hands': hands_data,
            'frame': frame  # Сохраняем кадр для визуализации
        })
        
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  Кадр {frame_idx}/{total_frames}")
    
    cap.release()
    mp_hands.close()
    print(f"Всего кадров: {len(all_frames_data)}")
    
    # Создаём скользящие окна и предсказываем
    print("Предсказание пошагово...")
    predictions = []
    
    # Сохраняем ключевые кадры для визуализации
    Path(output_frames_dir).mkdir(parents=True, exist_ok=True)
    
    for start in range(0, len(all_frames_data) - window_size + 1, step):
        window = all_frames_data[start:start + window_size]
        
        # Проверяем что есть руки
        windows_landmarks = []
        for f in window:
            if f['hands']:
                hand1 = f['hands'][0]
                hand2 = f['hands'][1] if len(f['hands']) > 1 else f['hands'][0]
                flat1 = [c for p in hand1 for c in p]
                flat2 = [c for p in hand2 for c in p]
                windows_landmarks.append(flat1 + flat2)
        
        if len(windows_landmarks) < window_size:
            continue
        
        # Предсказание
        x = torch.FloatTensor([windows_landmarks]).to(device)
        with torch.no_grad():
            output = model(x)
            prob = torch.softmax(output, dim=1)
            conf, pred = prob.max(1)
        
        label = idx_to_label[pred.item()]
        confidence = conf.item()
        mid_time = window[window_size // 2]['timestamp']
        
        if confidence >= min_conf and label != 'no_event':
            # Добавляем только если слово изменилось
            if not predictions or predictions[-1]['label'] != label:
                predictions.append({
                    'label': label,
                    'confidence': confidence,
                    'timestamp': mid_time,
                    'frame_idx': window[window_size // 2]['frame_idx']
                })
                
                # Сохраняем кадр с визуализацией
                frame_data = window[window_size // 2]
                vis_frame = frame_data['frame'].copy()
                vis_frame = draw_landmarks(vis_frame, frame_data['hands'])
                # Добавляем текст предсказания
                cv2.putText(vis_frame, f"{label} ({confidence:.0%})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(f"{output_frames_dir}/frame_{len(predictions):04d}_{mid_time:.1f}s.jpg", vis_frame)
    
    print(f"\nПредсказано: {len(predictions)} жестов")
    
    # Сохраняем результат
    output = {
        'video': video_path,
        'fps': fps,
        'predictions': predictions,
        'text': ' '.join([p['label'] for p in predictions])
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Сохранено: {output_json}")
    print(f"Кадры: {output_frames_dir}/")
    print(f"\nТекст: {output['text'][:300]}")
    
    # Печатаем пошагово
    print(f"\n=== Пошаговый перевод ===")
    for p in predictions[:30]:
        print(f"  {p['timestamp']:6.1f}с: {p['label']} ({p['confidence']:.0%})")
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Путь к видео')
    parser.add_argument('--model', default='translator_model_pytorch.pt')
    parser.add_argument('--output', default='step_by_step.json')
    parser.add_argument('--frames-dir', default='step_frames')
    parser.add_argument('--window-size', type=int, default=22)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--min-conf', type=float, default=0.4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model, idx_to_label = load_model(args.model, device)
    print(f"Классов: {len(idx_to_label)}")
    
    process_video_step_by_step(
        args.video, model, idx_to_label, device,
        args.output, args.frames_dir,
        args.window_size, args.step, args.min_conf
    )


if __name__ == '__main__':
    main()
