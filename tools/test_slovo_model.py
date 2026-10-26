"""
Тестирование pre-trained модели Slovo MViTv2 на наших данных.

Модель: MViTv2-small-32-2 (64.09% accuracy)
Вход: 32 кадра с интервалом 2 (64 кадра видео)
Выход: 1000 классов жестов русского ЖЯ
"""

import torch
import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple
import sys

# Добавляем slovo_repo в path
sys.path.insert(0, str(Path(__file__).parent.parent / 'slovo_repo'))
from constants import classes


class MViTv2Predictor:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Загрузка модели
        print(f"Загрузка модели из {model_path}...")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("✓ Модель загружена")
        
        # Параметры модели (из README)
        self.num_frames = 32
        self.frame_interval = 2
        self.input_size = 224
        
    def preprocess_video(self, video_path: str, start_frame: int = 0) -> Tuple[np.ndarray, List[int]]:
        """
        Извлекает 32 кадра из видео с интервалом 2.
        
        Returns:
            frames: np.ndarray shape (32, 224, 224, 3)
            frame_indices: список индексов кадров
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Выбираем 64 кадра (32 * 2) начиная с start_frame
        frame_indices = []
        for i in range(self.num_frames):
            idx = start_frame + i * self.frame_interval
            if idx < total_frames:
                frame_indices.append(idx)
        
        if len(frame_indices) < self.num_frames:
            print(f"⚠ Недостаточно кадров: {len(frame_indices)} < {self.num_frames}")
            # Дополняем последним кадром
            while len(frame_indices) < self.num_frames:
                frame_indices.append(frame_indices[-1])
        
        # Извлекаем кадры
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize to 224x224
            frame = cv2.resize(frame, (self.input_size, self.input_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Normalize to [0, 1]
        frames = np.array(frames, dtype=np.float32) / 255.0
        
        return frames, frame_indices
    
    def predict(self, frames: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Предсказывает класс жеста.
        
        Args:
            frames: np.ndarray shape (32, 224, 224, 3)
        
        Returns:
            predicted_class: str
            confidence: float
            top5: List of (class, confidence) tuples
        """
        # Convert to tensor: (1, 3, 32, 224, 224)
        # Model expects: (batch, channels, frames, height, width)
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            
        # Top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
        
        top5 = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            class_name = classes[idx.item()] if idx.item() < len(classes) else f"class_{idx.item()}"
            top5.append((class_name, prob.item()))
        
        predicted_class = top5[0][0]
        confidence = top5[0][1]
        
        return predicted_class, confidence, top5
    
    def predict_video(self, video_path: str, window_size: int = 64, step: int = 16) -> List[dict]:
        """
        Обрабатывает видео скользящим окном.
        
        Args:
            video_path: путь к видео
            window_size: размер окна в кадрах (64 = 32 * 2)
            step: шаг окна в кадрах
        
        Returns:
            predictions: список предсказаний с таймстемпами
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        print(f"Видео: {video_path}")
        print(f"Кадров: {total_frames}, FPS: {fps:.1f}")
        print(f"Окно: {window_size} кадров, шаг: {step} кадров")
        print()
        
        predictions = []
        start_frame = 0
        
        while start_frame + window_size <= total_frames:
            # Извлекаем кадры
            frames, frame_indices = self.preprocess_video(video_path, start_frame)
            
            # Предсказание
            predicted_class, confidence, top5 = self.predict(frames)
            
            # Таймстемп
            time_sec = start_frame / fps
            
            prediction = {
                'start_frame': start_frame,
                'end_frame': start_frame + window_size,
                'time': f"{time_sec:.2f}s",
                'predicted': predicted_class,
                'confidence': f"{confidence:.3f}",
                'top5': [(cls, f"{conf:.3f}") for cls, conf in top5]
            }
            
            predictions.append(prediction)
            
            print(f"[{time_sec:6.2f}s] {predicted_class:20s} ({confidence:.3f})")
            if len(top5) > 1:
                print(f"         Top-5: {', '.join([f'{c}({p:.2f})' for c, p in top5[:3]])}")
            
            start_frame += step
        
        return predictions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Тестирование Slovo MViTv2 модели')
    parser.add_argument('video', help='Путь к видео файлу')
    parser.add_argument('--model', default='models/slovo/mvit32-2.pt', help='Путь к модели')
    parser.add_argument('--output', help='Путь для сохранения JSON с предсказаниями')
    parser.add_argument('--window', type=int, default=64, help='Размер окна в кадрах')
    parser.add_argument('--step', type=int, default=16, help='Шаг окна в кадрах')
    
    args = parser.parse_args()
    
    # Загрузка модели
    predictor = MViTv2Predictor(args.model)
    
    # Обработка видео
    predictions = predictor.predict_video(args.video, args.window, args.step)
    
    print(f"\n✓ Всего предсказаний: {len(predictions)}")
    
    # Сохранение результатов
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"✓ Результаты сохранены в {args.output}")
    
    # Статистика
    unique_predictions = set(p['predicted'] for p in predictions)
    print(f"✓ Уникальных классов: {len(unique_predictions)}")
    print(f"✓ Классы: {', '.join(sorted(unique_predictions)[:20])}")


if __name__ == '__main__':
    main()
