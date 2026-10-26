"""
Демо-приложение: распознавание изолированных жестов RSL с помощью Easy_sign.
Видео → кадры → предсказания жестов → визуализация.
"""
import cv2
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


def load_class_list(path: Path) -> Dict[int, str]:
    """Загружает список классов из файла."""
    classes = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                idx, name = parts
                classes[int(idx)] = name
    return classes


def extract_clip_frames(video_path: Path, start_sec: float, duration_sec: float, target_frames: int = 32) -> np.ndarray:
    """
    Извлекает кадры из клипа видео.
    Возвращает [target_frames, 224, 224, 3] normalized.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)
    end_frame = min(end_frame, total_frames)
    
    # Равномерно сэмплируем target_frames кадров
    frame_indices = np.linspace(start_frame, end_frame - 1, target_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # Дополняем, если нужно
    while len(frames) < target_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Нормализация
    frames = np.array(frames, dtype=np.float32) / 255.0
    # [T, H, W, C] -> [C, T, H, W]
    frames = np.transpose(frames, (3, 0, 1, 2))
    # Batch dimension
    frames = np.expand_dims(frames, axis=0)
    
    return frames


def predict_gesture(
    sess: ort.InferenceSession,
    input_name: str,
    frames: np.ndarray,
    classes: Dict[int, str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Предсказывает жест для клипа.
    Возвращает top-K предсказаний: [(class_name, probability), ...]
    """
    outputs = sess.run(None, {input_name: frames})
    probs = outputs[0][0]  # [num_classes]
    
    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = [(classes.get(idx, f'unknown_{idx}'), float(probs[idx])) for idx in top_indices]
    
    return predictions


def process_video_sliding_window(
    video_path: Path,
    model_path: Path,
    classes_path: Path,
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    confidence_threshold: float = 0.5,
    output_path: Path = None,
):
    """
    Обрабатывает видео скользящим окном.
    Для каждого окна предсказывает жест.
    """
    # Загружаем модель
    print(f"Загружаю модель: {model_path}")
    sess = ort.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    
    # Загружаем классы
    classes = load_class_list(classes_path)
    print(f"Классов: {len(classes)}")
    
    # Видео info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    print(f"Видео: {video_path}")
    print(f"Длительность: {duration:.1f}с, FPS: {fps:.2f}")
    print(f"Окно: {window_sec}с, шаг: {step_sec}с")
    print()
    
    # Скользящее окно
    results = []
    start_sec = 0.0
    
    while start_sec + window_sec <= duration:
        # Извлекаем клип
        frames = extract_clip_frames(video_path, start_sec, window_sec)
        
        # Предсказание
        predictions = predict_gesture(sess, input_name, frames, classes, top_k=5)
        top_class, top_prob = predictions[0]
        
        result = {
            'start_sec': start_sec,
            'end_sec': start_sec + window_sec,
            'gesture': top_class,
            'confidence': top_prob,
            'top_predictions': predictions,
        }
        results.append(result)
        
        # Показываем, если уверенность высокая
        if top_prob >= confidence_threshold:
            print(f"{start_sec:.1f}-{start_sec+window_sec:.1f}с: {top_class} ({top_prob:.3f})")
        
        start_sec += step_sec
    
    print(f"\nОбработано {len(results)} окон")
    
    # Статистика
    gesture_counts = {}
    for r in results:
        if r['confidence'] >= confidence_threshold:
            g = r['gesture']
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
    
    print(f"\n=== Топ-20 жестов (confidence >= {confidence_threshold}) ===")
    for gesture, count in sorted(gesture_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"{gesture}: {count}")
    
    # Сохраняем результаты
    if output_path:
        output_data = {
            'video': str(video_path),
            'fps': fps,
            'duration': duration,
            'window_sec': window_sec,
            'step_sec': step_sec,
            'confidence_threshold': confidence_threshold,
            'num_windows': len(results),
            'results': results,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Демо: распознавание жестов RSL")
    parser.add_argument('video', type=Path, help='Путь к видео файлу')
    parser.add_argument('--model', type=Path, default=Path('models/easy_sign/S3D.onnx'))
    parser.add_argument('--classes', type=Path, default=Path('models/easy_sign/RSL_class_list.txt'))
    parser.add_argument('--window', type=float, default=2.0, help='Длина окна в секундах')
    parser.add_argument('--step', type=float, default=1.0, help='Шаг окна в секундах')
    parser.add_argument('--confidence', type=float, default=0.5, help='Порог уверенности')
    parser.add_argument('--output', type=Path, help='Путь для сохранения JSON')
    
    args = parser.parse_args()
    
    process_video_sliding_window(
        video_path=args.video,
        model_path=args.model,
        classes_path=args.classes,
        window_sec=args.window,
        step_sec=args.step,
        confidence_threshold=args.confidence,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
