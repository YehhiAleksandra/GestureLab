#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GestureLab PC — прототип этапа 0 ТЗ «Переводчик жестов».
Веб-камера -> MediaPipe Hands (2 кисти) + Face Detection.
Считает долю кадров с обеими кистями и с лицом, пишет сессии в JSON v2
(читается и v1 от Android/старых записей).

Клавиши (режим камеры):
  r  — старт/стоп записи сессии жеста (жест и исполнитель спросит в консоли)
  n  — то же для сессии «не жест» (обычные движения рук, фон для этапа 1)
  p  — переключить положение камеры
  q  — выход (идущая сессия сохраняется)

Примеры:
  python gesturelab.py --camera 0 --out sessions
  python gesturelab.py --input hands.mp4 --no-preview --out sessions   # offline-прогон
  python gesturelab.py --calibrate sessions                            # статистика положений
  python gesturelab.py --calibrate --input hands.mp4 --no-preview      # подбор порога детекции
"""

import argparse
import sys
import time
import tomllib
from pathlib import Path

import cv2

import gl_capture
import gl_detect
import gl_ui
from gl_session import (KIND_GESTURE, KIND_NOISE, POSITIONS, pct, save_session)
from summarize import aggregate, load_sessions, render

DEFAULT_CONFIG = {
    "positions": POSITIONS,
    "fps": 15.0,
    "max_session_seconds": 600.0,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "face_confidence": 0.5,
}


def load_config(path: Path) -> dict:
    conf = dict(DEFAULT_CONFIG)
    if path.exists():
        with open(path, "rb") as fh:
            conf.update(tomllib.load(fh))
    return conf


def run_loop(cap, args, out_dir: Path, detectors, interactive: bool, conf: dict):
    hands, face = detectors
    positions = conf["positions"]
    pos_idx = 0
    session = None          # метаданные текущей сессии (+ ключ elapsed для HUD)
    frames = []             # кадры текущей сессии
    start = 0.0
    totals = {"frames": 0, "both": 0, "face": 0}
    min_interval = 1.0 / max(conf["fps"], 1.0)
    last_record = 0.0
    win_start, win_frames, fps = time.time(), 0, 0.0
    show = not args.no_preview
    saved = None

    def stop_session(reason: str):
        nonlocal session, saved
        saved = save_session(out_dir, session, frames, totals)
        print(f"  -> сохранено {saved.name} ({reason}) | кадров {totals['frames']} | "
              f"обе кисти {pct(totals['both'], totals['frames']):.1f}% | "
              f"лицо {pct(totals['face'], totals['frames']):.1f}%")
        session = None

    def start_session(kind: str):
        nonlocal session, frames, totals, start, last_record
        if kind == KIND_GESTURE:
            gesture = input("Жест (например: да): ").strip() or "жест"
        else:
            gesture = input("Что делаешь (например: телефон): ").strip() or "фон"
        person = input("Исполнитель (инициалы): ").strip() or "-"
        session = {"gesture": gesture, "person": person,
                   "position": positions[pos_idx], "kind": kind, "elapsed": 0.0}
        frames = []
        totals = {"frames": 0, "both": 0, "face": 0}
        start = time.time()
        last_record = 0.0
        print(f"  -> запись «{kind}». Затем снова {('r' if kind == KIND_GESTURE else 'n')} — стоп.")

    if interactive:
        print("GestureLab PC. Клавиши: r — жест, n — «не жест», p — положение, q — выход")
    else:
        session = {"gesture": args.gesture, "person": args.person,
                   "position": args.position, "kind": KIND_GESTURE, "elapsed": 0.0}
        start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            if interactive:
                print("Кадр не прочитан (обрыв камеры или конец файла).")
            break
        if frame is None or frame.size == 0:
            continue  # неконсистентный кадр не должен ни считаться, ни ронять цикл
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        num_hands, has_face, landmarks_json, landmarks = gl_detect.process_frame(rgb, hands, face)
        rgb.flags.writeable = True

        totals["frames"] += 1
        if num_hands >= 2:
            totals["both"] += 1
        if has_face:
            totals["face"] += 1

        if session:
            session["elapsed"] = now - start
            if (now - last_record) >= min_interval:
                frames.append({
                    "t_ms": int((now - start) * 1000),
                    "hands": num_hands,
                    "face": has_face,
                    "hand_landmarks": landmarks_json,
                })
                last_record = now

        if show:
            gl_ui.draw_hud(frame, session, positions[pos_idx], totals, num_hands, fps,
                           landmarks, not args.no_preview_landmarks)
            cv2.imshow("GestureLab PC", frame)

        win_frames += 1
        if now - win_start >= 1.0:
            fps = win_frames / (now - win_start)
            win_frames, win_start = 0, now

        key = cv2.waitKey(1) & 0xFF if show else -1
        if interactive:
            if key == ord("q"):
                break
            elif key == ord("p"):
                pos_idx = (pos_idx + 1) % len(positions)
            elif key in (ord("r"), ord("n")) and session is None:
                start_session(KIND_GESTURE if key == ord("r") else KIND_NOISE)
            elif key in (ord("r"), ord("n")) and session is not None:
                stop_session("стоп по клавише")
        elif key == ord("q"):
            break

        if session and session["elapsed"] > conf["max_session_seconds"]:
            stop_session(f"авто-стоп по лимиту {conf['max_session_seconds']:.0f} с")

        if show and cv2.getWindowProperty("GestureLab PC", cv2.WND_PROP_VISIBLE) < 1:
            break

    if session is not None:
        stop_session("сохранено при выходе: обрыв камеры или конец файла")

    cap.release()
    if show:
        cv2.destroyAllWindows()
    return saved


def calibrate_sessions(folder: str) -> None:
    sessions, n_found, skipped = load_sessions(folder)
    if not n_found:
        sys.exit(f"В {folder} нет JSON-файлов.")
    print(f"Файлов: {n_found} (разобрано {len(sessions)})")
    stats = aggregate(sessions)
    lines, _ = render(stats, False)
    print("\n".join(lines))
    best = max(stats.items(), key=lambda kv: (kv[1][1] / kv[1][0] if kv[1][0] else 0.0))
    print(f"Рекомендуемое положение по доле кадров с обеими кистями: {best[0]}")
    print("Подбор min_detection_confidence по JSON невозможен: видео не сохраняется "
          "намеренно. Запусти: python gesturelab.py --calibrate --input <видео>")


def calibrate_video(path: str, conf: dict) -> None:
    print("Подбор порога детекции на видео (обе кисти / лицо, % кадров):")
    results = []
    for c in (0.3, 0.4, 0.5, 0.6, 0.7):
        cap = gl_capture.open_video(path)
        hands, face = gl_detect.build_detectors({**conf, "min_detection_confidence": c,
                                                 "min_tracking_confidence": c})
        n = both = face_n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            nh, hf, _, _ = gl_detect.process_frame(rgb, hands, face)
            rgb.flags.writeable = True
            n += 1
            both += nh >= 2
            face_n += hf
        cap.release()
        results.append((c, pct(both, n), pct(face_n, n)))
        print(f"  confidence {c:.1f}: обе кисти {pct(both, n):5.1f}% | лицо {pct(face_n, n):5.1f}%")
    best = max(results, key=lambda r: (r[1], -r[0]))
    print(f"Рекомендация: min_detection_confidence = {best[0]:.1f} "
          f"(максимум обеих кистей без завышения порога)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="индекс камеры (0 по умолчанию)")
    ap.add_argument("--input", help="видеофайл вместо камеры: offline-прогон без клавиш")
    ap.add_argument("--out", default="sessions", help="папка для JSON-сессий")
    ap.add_argument("--config", default="gesturelab.toml", help="файл конфигурации")
    ap.add_argument("--fps", type=float, help="частота записи кадров (перебивает конфиг)")
    ap.add_argument("--no-preview-landmarks", action="store_true", help="не рисовать скелет рук")
    ap.add_argument("--no-preview", action="store_true", help="не открывать окно (headless)")
    ap.add_argument("--gesture", default="offline", help="метка жеста для --input")
    ap.add_argument("--person", default="-", help="метка исполнителя для --input")
    ap.add_argument("--position", help="положение камеры для --input (по умолчанию из конфига)")
    ap.add_argument("--calibrate", nargs="?", const="", metavar="FOLDER",
                    help="папка сессий для статистики; с --input — подбор порога на видео")
    args = ap.parse_args()

    conf = load_config(Path(args.config))
    if args.fps is not None:
        conf["fps"] = args.fps
    if args.position is None:
        args.position = conf["positions"][0]

    if args.calibrate is not None:
        if args.input:
            calibrate_video(args.input, conf)
        else:
            calibrate_sessions(args.calibrate)
        return

    gl_capture.warn_if_non_ascii_path()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = gl_capture.open_video(args.input) if args.input else gl_capture.open_camera(args.camera)
    run_loop(cap, args, out_dir, gl_detect.build_detectors(conf), not args.input, conf)


if __name__ == "__main__":
    main()
