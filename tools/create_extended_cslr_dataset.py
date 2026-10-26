"""
Создание расширенного CSLR датасета с pseudo-labels от Easy_sign.
Прогоняет модель скользящим окном по всем Bible RSL видео.
"""
import cv2
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path
from tqdm import tqdm


def load_class_list(path: Path):
    """Загружает список классов."""
    classes = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                idx, name = parts
                classes[int(idx)] = name
    return classes


def extract_clip_frames(cap, start_frame, window_frames, target_frames=32):
    """Извлекает кадры из видео."""
    frame_indices = np.linspace(start_frame, start_frame + window_frames - 1, target_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    while len(frames) < target_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    frames_arr = np.array(frames, dtype=np.float32) / 255.0
    frames_arr = np.transpose(frames_arr, (3, 0, 1, 2))
    frames_arr = np.expand_dims(frames_arr, axis=0)
    
    return frames_arr


def process_video(video_path: Path, sess, input_name: str, classes: dict, window_sec=2.0, step_sec=1.0):
    """Обрабатывает видео скользящим окном."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    window_frames = int(window_sec * fps)
    step_frames = int(step_sec * fps)
    
    segments = []
    start_frame = 0
    
    pbar = tqdm(total=total_frames, desc=f"Обработка {video_path.name}")
    
    while start_frame + window_frames <= total_frames:
        # Извлекаем клип
        frames = extract_clip_frames(cap, start_frame, window_frames)
        
        # Инференс
        outputs = sess.run(None, {input_name: frames})
        probs = outputs[0][0]
        
        # Top-1 предсказание
        pred_idx = int(np.argmax(probs))
        pred_name = classes.get(pred_idx, f'unknown_{pred_idx}')
        pred_prob = float(probs[pred_idx])
        
        # Также извлекаем landmarks для CSLR
        # (используем уже обработанные landmarks JSON)
        
        segment = {
            'video_id': video_path.stem,
            'start_sec': start_frame / fps,
            'end_sec': (start_frame + window_frames) / fps,
            'start_frame': start_frame,
            'end_frame': start_frame + window_frames,
            'gesture': pred_name,
            'gesture_id': pred_idx,
            'confidence': pred_prob,
        }
        segments.append(segment)
        
        start_frame += step_frames
        pbar.update(step_frames)
    
    pbar.close()
    cap.release()
    
    return segments


def main():
    # Загружаем модель
    print("Загружаю Easy_sign модель...")
    sess = ort.InferenceSession('models/easy_sign/S3D.onnx')
    input_name = sess.get_inputs()[0].name
    
    classes = load_class_list(Path('models/easy_sign/RSL_class_list.txt'))
    print(f"Классов: {len(classes)}\n")
    
    # Видео
    videos = [
        Path('datasets/youtube_rsl/26RsAcQCVBA.mkv'),
        Path('datasets/youtube_rsl/9UdEIyOCQUY.mkv'),
        Path('datasets/youtube_rsl/Qcc3VBT4VT4.mkv'),
    ]
    
    # Параметры
    window_sec = 2.0
    step_sec = 1.0
    
    all_segments = []
    
    for video_path in videos:
        if not video_path.exists():
            print(f"Пропускаю {video_path} — не найдено")
            continue
        
        print(f"\n=== Обработка {video_path.name} ===")
        segments = process_video(video_path, sess, input_name, classes, window_sec, step_sec)
        all_segments.extend(segments)
        
        print(f"Создано {len(segments)} сегментов")
    
    # Статистика
    print(f"\n=== Итого ===")
    print(f"Всего сегментов: {len(all_segments)}")
    
    # Распределение по жестам
    gesture_counts = {}
    for seg in all_segments:
        g = seg['gesture']
        gesture_counts[g] = gesture_counts.get(g, 0) + 1
    
    print(f"Уникальных жестов: {len(gesture_counts)}")
    print(f"\nТоп-20 жестов:")
    for gesture, count in sorted(gesture_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {gesture}: {count}")
    
    # Сохраняем
    output_path = Path('datasets/youtube_rsl/cslr_dataset_extended.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_segments': len(all_segments),
            'window_sec': window_sec,
            'step_sec': step_sec,
            'segments': all_segments,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nДатасет сохранён: {output_path}")


if __name__ == '__main__':
    main()
