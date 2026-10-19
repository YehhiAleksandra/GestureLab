#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GestureLab PC — прототип этапа 0 ТЗ «Переводчик жестов».
Веб-камера -> MediaPipe Hands (2 кисти) + Face Detection.
Считает долю кадров с обеими кистями и с лицом, пишет сессии в JSON.
Формат JSON совместим с Android-приложением GestureLab v0.1.

Клавиши:
  r  — старт/стоп записи сессии (жест и исполнитель спросит в консоли)
  p  — переключить положение камеры (грудь/шнурок/подставка)
  q  — выход

Пример:  python gesturelab.py --camera 0 --out sessions
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

POSITIONS = ["грудь", "шнурок", "подставка"]


def pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-а-яА-Я]+", "_", s).strip("_") or "жест"


def landmarks_to_json(multi_hand_landmarks) -> list:
    """[[x0,y0,z0,x1,y1,z1,...x20,y20,z20], ...] — порядок как в Android-версии."""
    out = []
    if not multi_hand_landmarks:
        return out
    for hand in multi_hand_landmarks:
        flat = []
        for lm in hand.landmark:
            flat.extend((lm.x, lm.y, lm.z))
        out.append(flat)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="индекс камеры (0 по умолчанию)")
    ap.add_argument("--out", default="sessions", help="папка для JSON-сессий")
    ap.add_argument("--fps", type=float, default=15.0, help="частота записи кадров (цель ТЗ)")
    ap.add_argument("--no-preview-landmarks", action="store_true", help="не рисовать скелет рук")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        sys.exit("Не удалось открыть камеру. Проверь --camera (попробуй 0, 1, 2).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    pos_idx = 0
    session = None          # метаданные текущей сессии
    frames = []             # кадры текущей сессии
    start_ms = 0
    totals = {"frames": 0, "both": 0, "face": 0}
    min_interval = 1.0 / max(args.fps, 1.0)
    last_record = 0.0
    win_start, win_frames, fps = time.time(), 0, 0.0

    print("GestureLab PC. Клавиши: r — запись, p — положение, q — выход")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Кадр не прочитан, выход.")
            break
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res_h = hands.process(rgb)
        res_f = face.process(rgb)
        rgb.flags.writeable = True

        num_hands = len(res_h.multi_hand_landmarks or [])
        has_face = bool(res_f.detections)

        totals["frames"] += 1
        if num_hands >= 2:
            totals["both"] += 1
        if has_face:
            totals["face"] += 1

        if session and (now - last_record) >= min_interval:
            frames.append({
                "t_ms": int((now - start_ms) * 1000),
                "hands": num_hands,
                "face": has_face,
                "hand_landmarks": landmarks_to_json(res_h.multi_hand_landmarks),
            })
            last_record = now

        # --- отрисовка ---
        if not args.no_preview_landmarks and res_h.multi_hand_landmarks:
            for hand_lms in res_h.multi_hand_landmarks:
                draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        color = (0, 200, 0) if num_hands >= 2 else (0, 120, 255)
        cv2.rectangle(frame, (8, 8), (632, 118), color, 2)

        if session:
            rec_s = int(now - start_ms)
            line0 = f"ЗАПИСЬ {rec_s//60:02d}:{rec_s%60:02d} | жест: {session['gesture']} | {session['position']}"
        else:
            line0 = f"Позиция: {POSITIONS[pos_idx]} | клавиша r — начать запись, p — положение"
        line1 = (f"Кисти обе: {pct(totals['both'], totals['frames']):5.1f}% | "
                 f"Лицо: {pct(totals['face'], totals['frames']):5.1f}% | "
                 f"Кадров: {totals['frames']} | ~{fps:4.1f} к/с | рук в кадре: {num_hands}")
        cv2.putText(frame, line0, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, line1, (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if not session:
            cv2.putText(frame, "Эталон критерия: обе кисти >= 90% кадров", (14, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("GestureLab PC", frame)

        win_frames += 1
        if now - win_start >= 1.0:
            fps = win_frames / (now - win_start)
            win_frames, win_start = 0, now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            pos_idx = (pos_idx + 1) % len(POSITIONS)
        elif key == ord("r"):
            if session is None:
                gesture = input("Жест (например: да): ").strip() or "жест"
                person = input("Исполнитель (инициалы): ").strip() or "-"
                session = {"gesture": gesture, "person": person,
                           "position": POSITIONS[pos_idx]}
                frames = []
                totals = {"frames": 0, "both": 0, "face": 0}
                start_ms = time.time()
                last_record = 0.0
                print("  -> запись. Покажи жест, затем снова r — стоп.")
            else:
                data = {
                    "version": 1,
                    "gesture": session["gesture"],
                    "position": session["position"],
                    "person": session["person"],
                    "frames": frames,
                    "summary": {
                        "frames": totals["frames"],
                        "both_hands_pct": round(pct(totals["both"], totals["frames"]), 2),
                        "face_pct": round(pct(totals["face"], totals["frames"]), 2),
                    },
                }
                fname = f"{int(time.time()*1000)}_{safe_name(session['gesture'])}_{session['position']}.json"
                (out_dir / fname).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                print(f"  -> сохранено {fname} | кадров {totals['frames']} | "
                      f"обе кисти {pct(totals['both'], totals['frames']):.1f}% | "
                      f"лицо {pct(totals['face'], totals['frames']):.1f}%")
                session = None

        if cv2.getWindowProperty("GestureLab PC", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
