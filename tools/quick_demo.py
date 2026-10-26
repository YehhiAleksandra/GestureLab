"""
Быстрое демо: распознавание жестов на коротких клипах видео.
"""
import cv2
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path


def main():
    # Загружаем модель
    print('Загружаю Easy_sign модель...')
    sess = ort.InferenceSession('models/easy_sign/S3D.onnx')
    input_name = sess.get_inputs()[0].name
    
    # Загружаем классы
    classes = {}
    with open('models/easy_sign/RSL_class_list.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                idx, name = parts
                classes[int(idx)] = name
    
    print(f'Классов: {len(classes)}\n')
    
    # Видео
    video_path = Path('datasets/youtube_rsl/9UdEIyOCQUY.mkv')
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    cap.release()
    
    print(f'Видео: {video_path.name}')
    print(f'Длительность: {duration:.1f}с ({duration/60:.1f} мин)\n')
    
    # Обрабатываем каждые 10 секунд (быстрое демо)
    window_sec = 2.0
    step_sec = 10.0
    
    results = []
    start_sec = 0.0
    
    print('Обработка клипов:')
    while start_sec + window_sec <= min(duration, 60.0):  # Только первая минута
        # Извлекаем клип
        cap = cv2.VideoCapture(str(video_path))
        target_frames = 32
        window_frames = int(window_sec * fps)
        frame_indices = np.linspace(
            int(start_sec * fps),
            int((start_sec + window_sec) * fps) - 1,
            target_frames,
            dtype=int
        )
        
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
        
        if len(frames) < target_frames:
            frames += [frames[-1]] * (target_frames - len(frames))
        
        # Нормализация
        frames_arr = np.array(frames, dtype=np.float32) / 255.0
        frames_arr = np.transpose(frames_arr, (3, 0, 1, 2))
        frames_arr = np.expand_dims(frames_arr, axis=0)
        
        # Инференс
        outputs = sess.run(None, {input_name: frames_arr})
        probs = outputs[0][0]
        
        # Top-1
        pred_idx = int(np.argmax(probs))
        pred_name = classes.get(pred_idx, f'unknown_{pred_idx}')
        pred_prob = float(probs[pred_idx])
        
        results.append({
            'start_sec': start_sec,
            'end_sec': start_sec + window_sec,
            'gesture': pred_name,
            'confidence': pred_prob,
        })
        
        print(f'  {start_sec:.0f}-{start_sec+window_sec:.0f}с: {pred_name} ({pred_prob:.3f})')
        
        start_sec += step_sec
    
    print(f'\nОбработано {len(results)} клипов')
    
    # Сохраняем
    output_path = Path('datasets/youtube_rsl/demo_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'video': video_path.name,
            'duration': duration,
            'results': results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f'Результаты сохранены: {output_path}')
    
    # Статистика
    print('\n=== Распознанные жесты ===')
    for r in results:
        print(f"{r['start_sec']:.0f}-{r['end_sec']:.0f}с: {r['gesture']} ({r['confidence']:.3f})")


if __name__ == '__main__':
    main()
