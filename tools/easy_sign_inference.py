"""
Инференс Easy_sign модели на Bible RSL видео.
Модель S3D принимает [1, 3, 32, 224, 224] — 32 кадра 224x224 RGB.
Скользящее окно по видео, классификация каждого окна.
"""
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path
from tqdm import tqdm


def load_class_list(path):
    """Загружает список классов из файла."""
    classes = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                idx, name = parts
                classes[int(idx)] = name
    return classes


def extract_window_frames(video_path, start_frame, window_frames, target_frames=32):
    """
    Извлекает ровно target_frames кадров из окна видео.
    Равномерно сэмплирует из диапазона [start_frame, start_frame + window_frames].
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    # Равномерно распределяем target_frames кадров внутри окна
    frame_indices = np.linspace(start_frame, start_frame + window_frames - 1, target_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Если не удалось прочитать, используем последний успешный
            if frames:
                frames.append(frames[-1])
            continue
        # Resize to 224x224
        frame = cv2.resize(frame, (224, 224))
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # Гарантируем ровно target_frames
    if len(frames) < target_frames:
        # Дополняем последним кадром
        frames += [frames[-1]] * (target_frames - len(frames))

    # Нормализация [0, 255] -> [0, 1]
    frames = np.array(frames, dtype=np.float32) / 255.0
    # Транспонирование [T, H, W, C] -> [C, T, H, W]
    frames = np.transpose(frames, (3, 0, 1, 2))
    # Добавляем batch dimension
    frames = np.expand_dims(frames, axis=0)

    return frames


def sliding_window_inference(video_path, model_path, window_sec=2.0, step_sec=1.0):
    """
    Скользящее окно по видео с инференсом.
    window_sec: длина окна в секундах
    step_sec: шаг окна в секундах
    """
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    window_frames = int(window_sec * video_fps)
    step_frames = int(step_sec * video_fps)

    results = []
    start_frame = 0

    pbar = tqdm(total=total_frames, desc="Инференс")
    while start_frame + window_frames <= total_frames:
        # Извлекаем окно кадров (ровно 32 кадра)
        frames = extract_window_frames(
            video_path,
            start_frame,
            window_frames,
            target_frames=32
        )

        # Инференс
        outputs = sess.run(None, {input_name: frames})
        probs = outputs[0][0]  # [num_classes]
        pred_class = int(np.argmax(probs))
        pred_prob = float(probs[pred_class])

        results.append({
            'start_frame': start_frame,
            'start_sec': start_frame / video_fps,
            'end_sec': (start_frame + window_frames) / video_fps,
            'class_id': pred_class,
            'probability': pred_prob
        })

        start_frame += step_frames
        pbar.update(step_frames)

    pbar.close()
    return results, video_fps, total_frames


def main():
    parser = argparse.ArgumentParser(description="Easy_sign инференс на Bible RSL видео")
    parser.add_argument('video', type=Path, help='Путь к видео файлу')
    parser.add_argument('--model', type=Path, default=Path('models/easy_sign/S3D.onnx'),
                        help='Путь к ONNX модели')
    parser.add_argument('--classes', type=Path, default=Path('models/easy_sign/RSL_class_list.txt'),
                        help='Путь к списку классов')
    parser.add_argument('--window', type=float, default=2.0, help='Длина окна в секундах')
    parser.add_argument('--step', type=float, default=1.0, help='Шаг окна в секундах')
    parser.add_argument('--output', type=Path, help='Путь для сохранения JSON результатов')
    parser.add_argument('--top-k', type=int, default=5, help='Показать top-K предсказаний')

    args = parser.parse_args()

    # Загружаем список классов
    classes = load_class_list(args.classes)
    print(f"Загружено {len(classes)} классов")

    # Инференс
    print(f"Обработка видео: {args.video}")
    results, video_fps, total_frames = sliding_window_inference(
        args.video, args.model, args.window, args.step
    )

    print(f"Видео: {total_frames} кадров, {video_fps:.2f} FPS")
    print(f"Получено {len(results)} предсказаний")

    # Добавляем имена классов
    for r in results:
        r['class_name'] = classes.get(r['class_id'], f"unknown_{r['class_id']}")

    # Показываем результаты
    print("\n=== Результаты (первые 20 окон) ===")
    for r in results[:20]:
        print(f"{r['start_sec']:.1f}-{r['end_sec']:.1f}с: {r['class_name']} ({r['probability']:.3f})")

    # Статистика по классам
    class_counts = {}
    for r in results:
        name = r['class_name']
        class_counts[name] = class_counts.get(name, 0) + 1

    print("\n=== Топ-20 классов ===")
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"{name}: {count} окон ({count/len(results)*100:.1f}%)")

    # Сохраняем результаты
    if args.output:
        output_data = {
            'video': str(args.video),
            'fps': video_fps,
            'total_frames': total_frames,
            'window_sec': args.window,
            'step_sec': args.step,
            'predictions': results
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены: {args.output}")


if __name__ == '__main__':
    main()
