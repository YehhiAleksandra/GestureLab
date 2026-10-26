"""
Автоматическая разметка YouTube курса с использованием Slovo MViTv2 модели.

Скрипт:
1. Обрабатывает видео скользящим окном
2. Предсказывает жесты с помощью pre-trained модели
3. Сохраняет разметку в формате JSON для ручной проверки
4. Создаёт видео с визуализацией предсказаний
"""

import cv2
import numpy as np
import torch
import json
from pathlib import Path
from typing import List, Tuple
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'slovo_repo'))
from constants import classes


class AutoAnnotator:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {self.device}")
        
        print(f"Загрузка модели {model_path}...")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("✓ Модель загружена")
        
        self.num_frames = 32
        self.frame_interval = 2
        self.input_size = 224
        
    def extract_frames(self, video_path: str, start_frame: int) -> np.ndarray:
        """Извлекает 32 кадра с интервалом 2."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        for i in range(self.num_frames):
            idx = start_frame + i * self.frame_interval
            if idx >= total_frames:
                idx = total_frames - 1
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (self.input_size, self.input_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frames.append(frame)
        
        cap.release()
        return np.array(frames, dtype=np.float32) / 255.0
    
    def predict(self, frames: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Предсказывает класс жеста."""
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
        
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
        
        top5 = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            class_name = classes[idx.item()] if idx.item() < len(classes) else f"class_{idx.item()}"
            top5.append((class_name, prob.item()))
        
        return top5[0][0], top5[0][1], top5
    
    def annotate_video(self, video_path: str, output_json: str, 
                      window_size: int = 64, step: int = 16, 
                      confidence_threshold: float = 0.3) -> List[dict]:
        """
        Автоматически размечает видео.
        
        Args:
            video_path: путь к видео
            output_json: путь для сохранения JSON
            window_size: размер окна в кадрах
            step: шаг окна
            confidence_threshold: минимальная уверенность для сохранения
        
        Returns:
            annotations: список разметок
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        print(f"\nВидео: {Path(video_path).name}")
        print(f"Кадров: {total_frames}, FPS: {fps:.1f}, Длительность: {total_frames/fps:.1f}с")
        print(f"Окно: {window_size}, шаг: {step}, порог: {confidence_threshold}")
        print()
        
        annotations = []
        start_frame = 0
        last_gesture = None
        gesture_start = None
        
        while start_frame + window_size <= total_frames:
            frames = self.extract_frames(video_path, start_frame)
            predicted, confidence, top5 = self.predict(frames)
            
            time_sec = start_frame / fps
            
            # Сохраняем только если уверенность выше порога
            if confidence >= confidence_threshold:
                # Проверяем, изменился ли жест
                if predicted != last_gesture:
                    # Сохраняем предыдущий жест
                    if last_gesture is not None and gesture_start is not None:
                        end_time = (start_frame - step) / fps
                        if end_time > gesture_start:
                            annotations.append({
                                'start_time': gesture_start,
                                'end_time': end_time,
                                'text': last_gesture,
                                'confidence': f"{last_confidence:.3f}",
                                'source': 'auto',
                                'model': 'slovo-mvitv2',
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    # Начинаем новый жест
                    gesture_start = time_sec
                    last_gesture = predicted
                    last_confidence = confidence
                
                print(f"[{time_sec:6.2f}s] {predicted:20s} ({confidence:.3f})")
            
            start_frame += step
        
        # Сохраняем последний жест
        if last_gesture is not None and gesture_start is not None:
            end_time = total_frames / fps
            if end_time > gesture_start:
                annotations.append({
                    'start_time': gesture_start,
                    'end_time': end_time,
                    'text': last_gesture,
                    'confidence': f"{last_confidence:.3f}",
                    'source': 'auto',
                    'model': 'slovo-mvitv2',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Сохраняем результаты
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Сохранено {len(annotations)} разметок в {output_json}")
        
        # Статистика
        unique_gestures = set(a['text'] for a in annotations)
        print(f"✓ Уникальных жестов: {len(unique_gestures)}")
        print(f"✓ Жесты: {', '.join(sorted(unique_gestures)[:20])}")
        
        return annotations


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Автоматическая разметка видео')
    parser.add_argument('video', help='Путь к видео')
    parser.add_argument('--model', default='models/slovo/mvit32-2.pt', help='Путь к модели')
    parser.add_argument('--output', help='Путь для JSON (по умолчанию: <video>.json)')
    parser.add_argument('--window', type=int, default=64, help='Размер окна')
    parser.add_argument('--step', type=int, default=16, help='Шаг окна')
    parser.add_argument('--threshold', type=float, default=0.3, help='Порог уверенности')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = str(Path(args.video).with_suffix('.json'))
    
    annotator = AutoAnnotator(args.model)
    annotator.annotate_video(
        args.video, 
        args.output,
        args.window,
        args.step,
        args.threshold
    )


if __name__ == '__main__':
    main()
