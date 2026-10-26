"""
Генерация финального видео перевода с визуализацией и озвучкой.

Берёт автоматическую разметку и создаёт видео с:
1. Визуализацией рук (landmarks)
2. Субтитрами с переводом
3. Озвучкой через edge-tts
"""

import json
import cv2
import numpy as np
from pathlib import Path
import edge_tts
import asyncio
import tempfile
from typing import List, Dict


def draw_hand_landmarks(frame: np.ndarray, landmarks: List[float], 
                       color: tuple = (0, 255, 0), thickness: int = 2):
    """Рисует landmarks рук на кадре."""
    if len(landmarks) != 126:
        return frame
    
    h, w = frame.shape[:2]
    
    # Разделяем на две руки
    hand1 = landmarks[:63]
    hand2 = landmarks[63:]
    
    # Рисуем точки
    for i in range(21):
        # Hand 1
        x1, y1, z1 = hand1[i*3:(i+1)*3]
        px1, py1 = int(x1 * w), int(y1 * h)
        cv2.circle(frame, (px1, py1), 3, color, -1)
        
        # Hand 2
        x2, y2, z2 = hand2[i*3:(i+1)*3]
        px2, py2 = int(x2 * w), int(y2 * h)
        cv2.circle(frame, (px2, py2), 3, color, -1)
    
    # Соединения
    connections = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
                   (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
                   (13,17),(17,18),(18,19),(19,20), (0,17)]
    
    for conn in connections:
        i, j = conn
        
        # Hand 1
        x1, y1, _ = hand1[i*3:(i+1)*3]
        x2, y2, _ = hand1[j*3:(j+1)*3]
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        cv2.line(frame, pt1, pt2, color, thickness)
        
        # Hand 2
        x1, y1, _ = hand2[i*3:(i+1)*3]
        x2, y2, _ = hand2[j*3:(j+1)*3]
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        cv2.line(frame, pt1, pt2, color, thickness)
    
    return frame


def add_subtitle(frame: np.ndarray, text: str, confidence: float, 
                position: tuple = (10, 50)):
    """Добавляет субтитры на кадр."""
    h, w = frame.shape[:2]
    
    # Фон для текста
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, position[1] - 30), (w, position[1] + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Текст
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (position[0], position[1]), 
               font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Уверенность
    conf_text = f"{confidence:.2f}"
    cv2.putText(frame, conf_text, (w - 100, position[1]), 
               font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame


async def generate_audio(text: str, output_path: str, voice: str = "ru-RU-SvetlanaNeural"):
    """Генерирует аудио из текста."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def create_translation_video(annotation_json: str, landmarks_json: str,
                            output_video: str, output_audio: str = None):
    """Создаёт финальное видео перевода."""
    
    # Загрузка разметки
    print(f"Загрузка разметки: {annotation_json}")
    with open(annotation_json, 'r', encoding='utf-8') as f:
        gestures = json.load(f)
    
    print(f"Жестов: {len(gestures)}")
    
    # Загрузка landmarks
    print(f"Загрузка landmarks: {landmarks_json}")
    with open(landmarks_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    frames_data = data.get('frames', data) if isinstance(data, dict) else data
    print(f"Кадров: {len(frames_data)}")
    
    # Видео параметры
    fps = 25.0
    width, height = 1280, 720
    
    # Создание видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"\nСоздание видео: {output_video}")
    
    # Обработка кадров
    current_gesture_idx = 0
    text_for_audio = []
    
    for frame_idx in range(len(frames_data)):
        # Извлекаем landmarks
        frame_data = frames_data[frame_idx]
        landmarks = None
        
        hands = frame_data.get("hands", [])
        if hands:
            hand1 = hands[0]
            hand2 = hands[1] if len(hands) > 1 else hand1
            flat1 = [coord for point in hand1 for coord in point]
            flat2 = [coord for point in hand2 for coord in point]
            landmarks = flat1 + flat2
        
        # Создаём кадр
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Серый фон
        
        # Рисуем landmarks
        if landmarks:
            frame = draw_hand_landmarks(frame, landmarks, color=(0, 255, 0))
        
        # Проверяем, активен ли текущий жест
        if current_gesture_idx < len(gestures):
            gesture = gestures[current_gesture_idx]
            start_frame = int(gesture['start_frame'])
            end_frame = int(gesture['end_frame'])
            
            if start_frame <= frame_idx < end_frame:
                # Активный жест - показываем субтитры
                frame = add_subtitle(frame, gesture['label'], gesture['confidence'])
                
                # Добавляем в текст для озвучки (один раз)
                if frame_idx == start_frame:
                    text_for_audio.append(gesture['label'])
            
            elif frame_idx >= end_frame:
                current_gesture_idx += 1
        
        # Добавляем номер кадра
        cv2.putText(frame, f"Frame: {frame_idx}/{len(frames_data)}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out.write(frame)
        
        if (frame_idx + 1) % 1000 == 0:
            print(f"  Обработано {frame_idx + 1}/{len(frames_data)} кадров")
    
    out.release()
    print(f"✓ Видео создано: {output_video}")
    
    # Генерация аудио
    if output_audio and text_for_audio:
        full_text = ' '.join(text_for_audio)
        print(f"\nГенерация аудио: {output_audio}")
        print(f"Текст: {full_text[:100]}...")
        
        asyncio.run(generate_audio(full_text, output_audio))
        print(f"✓ Аудио создано: {output_audio}")
    
    # Статистика
    print(f"\n=== Статистика ===")
    print(f"Жестов распознано: {len(gestures)}")
    print(f"Уникальных: {len(set(g['label'] for g in gestures))}")
    print(f"Кадров обработано: {len(frames_data)}")
    print(f"Длительность видео: {len(frames_data)/fps:.1f}с")
    
    return output_video, output_audio


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True, help='JSON с разметкой')
    parser.add_argument('--landmarks', required=True, help='JSON с landmarks')
    parser.add_argument('--output-video', required=True, help='Выходное видео')
    parser.add_argument('--output-audio', help='Выходное аудио (опционально)')
    
    args = parser.parse_args()
    
    create_translation_video(
        args.annotation,
        args.landmarks,
        args.output_video,
        args.output_audio
    )
