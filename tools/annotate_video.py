"""
Инструмент для интерактивной разметки видео жестового языка.

Использование:
    python tools/annotate_video.py <путь_к_видео>

Управление:
    Пробел - play/pause
    Стрелка влево/вправо - перемотка на 1 секунду
    Ctrl + стрелка влево/вправо - перемотка на 5 секунд
    M - установить маркер начала жеста
    N - установить маркер конца жеста
    T - ввести текст перевода для текущего сегмента
    S - сохранить текущую разметку
    Q - выход
    R - сбросить текущий сегмент

Разметка сохраняется в JSON файл рядом с видео.
"""

import cv2
import json
import sys
from pathlib import Path
from datetime import datetime


class VideoAnnotator:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.annotation_path = self.video_path.with_suffix('.json')
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        self.annotations = self.load_annotations()
        self.current_segment = {'start': None, 'end': None, 'text': ''}
        
        self.playing = False
        self.current_frame = 0
        
        print(f"Видео: {self.video_path.name}")
        print(f"FPS: {self.fps:.1f}, Кадров: {self.total_frames}, Длительность: {self.duration:.1f}с")
        print(f"Загружено разметок: {len(self.annotations)}")
        print("\nУправление:")
        print("  Пробел - play/pause")
        print("  ←/→ - перемотка ±1с")
        print("  Ctrl+←/→ - перемотка ±5с")
        print("  M - начало жеста")
        print("  N - конец жеста")
        print("  T - ввести текст")
        print("  S - сохранить")
        print("  R - сбросить сегмент")
        print("  Q - выход")
        print()
    
    def load_annotations(self):
        if self.annotation_path.exists():
            with open(self.annotation_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_annotations(self):
        with open(self.annotation_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        print(f"✓ Сохранено {len(self.annotations)} разметок в {self.annotation_path.name}")
    
    def get_time_str(self, frame_num):
        seconds = frame_num / self.fps
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:05.2f}"
    
    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        
        # Текущее время
        time_str = self.get_time_str(self.current_frame)
        cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Текущий сегмент
        if self.current_segment['start'] is not None:
            start_time = self.get_time_str(self.current_segment['start'])
            end_time = self.get_time_str(self.current_segment['end']) if self.current_segment['end'] else '???'
            text = self.current_segment['text']
            
            cv2.putText(frame, f"Segment: {start_time} - {end_time}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if text:
                cv2.putText(frame, f"Text: {text}", (10, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Существующие разметки
        for i, ann in enumerate(self.annotations[-5:]):  # Последние 5
            y = h - 30 - i * 25
            time_range = f"{self.get_time_str(int(ann['start_frame']))}-{self.get_time_str(int(ann['end_frame']))}"
            text = ann['text'][:30]
            cv2.putText(frame, f"{time_range}: {text}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Прогресс-бар
        progress = self.current_frame / self.total_frames
        cv2.rectangle(frame, (0, h-5), (int(w * progress), h), (0, 255, 0), -1)
        
        return frame
    
    def set_marker_start(self):
        self.current_segment['start'] = self.current_frame
        print(f"✓ Начало жеста: {self.get_time_str(self.current_frame)}")
    
    def set_marker_end(self):
        if self.current_segment['start'] is None:
            print("✗ Сначала установите начало (M)")
            return
        self.current_segment['end'] = self.current_frame
        print(f"✓ Конец жеста: {self.get_time_str(self.current_frame)}")
    
    def set_text(self):
        if self.current_segment['start'] is None or self.current_segment['end'] is None:
            print("✗ Сначала установите начало (M) и конец (N)")
            return
        
        text = input("Введите текст перевода: ").strip()
        if text:
            self.current_segment['text'] = text
            print(f"✓ Текст: {text}")
    
    def save_segment(self):
        if self.current_segment['start'] is None or self.current_segment['end'] is None:
            print("✗ Сначала установите начало (M) и конец (N)")
            return
        if not self.current_segment['text']:
            print("✗ Сначала введите текст (T)")
            return
        
        annotation = {
            'start_frame': self.current_segment['start'],
            'end_frame': self.current_segment['end'],
            'start_time': self.current_segment['start'] / self.fps,
            'end_time': self.current_segment['end'] / self.fps,
            'text': self.current_segment['text'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.annotations.append(annotation)
        self.save_annotations()
        self.reset_segment()
    
    def reset_segment(self):
        self.current_segment = {'start': None, 'end': None, 'text': ''}
        print("✓ Сегмент сброшен")
    
    def run(self):
        cv2.namedWindow('Annotation', cv2.WINDOW_NORMAL)
        
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = self.draw_overlay(frame)
            cv2.imshow('Annotation', frame)
            
            key = cv2.waitKeyEx(30)  # ~33 FPS
            
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # Space - play/pause
                self.playing = not self.playing
                print("▶ Play" if self.playing else "⏸ Pause")
            elif key == 2424832:  # Left arrow
                self.current_frame = max(0, self.current_frame - int(self.fps))
            elif key == 2555904:  # Right arrow
                self.current_frame = min(self.total_frames - 1, self.current_frame + int(self.fps))
            elif key == 2490368:  # Ctrl+Left
                self.current_frame = max(0, self.current_frame - int(5 * self.fps))
            elif key == 2621440:  # Ctrl+Right
                self.current_frame = min(self.total_frames - 1, self.current_frame + int(5 * self.fps))
            elif key == ord('m') or key == ord('M'):
                self.set_marker_start()
            elif key == ord('n') or key == ord('N'):
                self.set_marker_end()
            elif key == ord('t') or key == ord('T'):
                self.set_text()
            elif key == ord('s') or key == ord('S'):
                self.save_segment()
            elif key == ord('r') or key == ord('R'):
                self.reset_segment()
            
            if self.playing:
                self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nИтого разметок: {len(self.annotations)}")
        if self.annotations:
            self.save_annotations()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python tools/annotate_video.py <путь_к_видео>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    annotator = VideoAnnotator(video_path)
    annotator.run()
