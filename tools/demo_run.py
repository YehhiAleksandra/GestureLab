#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Виртуальный прогон GestureLab без камеры: скриншоты HUD и offline-детекция.

Примеры:
  python tools/demo_run.py                    # состояния HUD -> reports/img/*.png
  python tools/demo_run.py --video            # + синтетическое видео и прогон --input
  python tools/demo_run.py --image foto.jpg   # настоящая детекция на одном кадре

Людей в синтетических кадрах нет, видео в репозиторий не пишется (приватность
этапа 0): ролик и JSON демо-сессии создаются во временной папке.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import gesturelab                      # noqa: E402
import gl_detect                       # noqa: E402
import gl_ui                           # noqa: E402
from gl_session import KIND_GESTURE, KIND_NOISE  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "fix_both_hands.json"


def imread_u(path: Path):
    """cv2.imread не умеет не-ASCII пути (B22-семейство) — читаем через imdecode."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path: Path, img) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"не удалось закодировать {path.name}")
    buf.tofile(str(path))
    return path


class Point:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z

    def HasField(self, name):
        # drawing_utils отбраковывает точки с visibility/presence ниже порога; у нас их нет
        return False


class FakeHand:
    """Утиная замена proto MediaPipe: drawing_utils нужен только .landmark[x,y,z]."""

    def __init__(self, points):
        self.landmark = [Point(x, y) for x, y in points]


def hand_points(cx: float, cy: float, scale: float, mirror: bool = False) -> list:
    """21 точка кисти в порядке MediaPipe: 0 запястье, 1-4 большой, 5-8, 9-12, 13-16, 17-20.

    Координаты нормализованные (0..1), y вниз; пальцы — веер от сустава к суставу.
    """
    # (x сустава, y сустава, наклон в градусах от вертикали, длина пальца)
    fingers = ((-0.34, 0.10, -62, 0.44), (-0.13, -0.02, -14, 0.62),
               (0.02, -0.06, -2, 0.68), (0.16, -0.04, 12, 0.62),
               (0.28, 0.03, 26, 0.48))
    pts = [(cx, cy + 0.30 * scale)]
    for kx, ky, tilt, length in fingers:
        rad = math.radians(tilt)
        for frac in (0.0, 0.35, 0.68, 1.0):
            r = length * frac * scale
            pts.append((cx + kx * scale + r * math.sin(rad), cy + ky * scale - r * math.cos(rad)))
    if mirror:
        pts = [(2 * cx - x, y) for x, y in pts]
    return pts


def base_frame(w: int = 1280, h: int = 720):
    """Фон: вертикальный градиент и сетка — видно, что кадр синтетический."""
    grad = np.linspace(38, 92, h, dtype=np.uint8)[:, None].repeat(w, axis=1)
    frame = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
    for x in range(0, w, 80):
        cv2.line(frame, (x, 0), (x, h), (58, 58, 58), 1)
    for y in range(0, h, 80):
        cv2.line(frame, (0, y), (w, y), (58, 58, 58), 1)
    return frame


def fixture_hands() -> list:
    """Ландмарки из фикстуры (значения синтетические) — проверка формата, не вида."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    middle = data["frames"][len(data["frames"]) // 2]
    return [FakeHand([(v[0], v[1]) for v in
                      (flat[i:i + 3] for i in range(0, 63, 3))])
            for flat in middle["hand_landmarks"]]


def render_hud_states(out_dir: Path) -> list:
    """Состояния интерфейса: покой, запись жеста, запись «не жест», нет кистей."""
    two_hands = [FakeHand(hand_points(0.33, 0.62, 0.42)),
                 FakeHand(hand_points(0.67, 0.62, 0.42, mirror=True))]
    totals = {"frames": 120, "both": 111, "face": 96}
    cases = [
        ("hud_idle.png", None, "грудь", two_hands, totals, 14.8),
        ("hud_record_gesture.png",
         {"kind": KIND_GESTURE, "gesture": "да", "position": "грудь", "elapsed": 12.4},
         "грудь", two_hands, totals, 14.6),
        ("hud_record_noise.png",
         {"kind": KIND_NOISE, "gesture": "телефон", "position": "шнурок", "elapsed": 7.0},
         "шнурок", two_hands, {"frames": 64, "both": 61, "face": 12}, 15.1),
        ("hud_no_hands.png", None, "подставка", [], {"frames": 90, "both": 0, "face": 88}, 13.2),
        ("hud_zero_frames.png", None, "грудь", [], {"frames": 0, "both": 0, "face": 0}, 0.0),
        ("hud_fixture_landmarks.png", None, "грудь", fixture_hands(),
         {"frames": 30, "both": 30, "face": 30}, 12.0),
    ]
    saved = []
    for name, session, pos_label, landmarks, tot, fps in cases:
        frame = base_frame()
        gl_ui.draw_hud(frame, session, pos_label, tot, len(landmarks), fps,
                       landmarks, bool(landmarks))
        saved.append(imwrite_u(out_dir / name, frame))
    return saved


def detect_on_frame(frame, conf: dict, out_path: Path, session=None):
    """Настоящий MediaPipe на одном кадре -> статистика и аннотированный PNG."""
    hands, face = gl_detect.build_detectors(conf)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    num_hands, has_face, _, landmarks = gl_detect.process_frame(rgb, hands, face)
    rgb.flags.writeable = True
    hands.close()
    face.close()
    totals = {"frames": 1, "both": int(num_hands >= 2), "face": int(has_face)}
    annotated = frame.copy()
    gl_ui.draw_hud(annotated, session, "один кадр", totals, num_hands, 0.0,
                   landmarks, True)
    imwrite_u(out_path, annotated)
    print(f"  кадр {out_path.name}: кистей {num_hands}, лицо {'да' if has_face else 'нет'}")
    return num_hands, has_face


def make_demo_video(path: Path, seconds: float = 6.0, fps: int = 15) -> Path:
    """Синтетическое видео (движущиеся прямоугольники) для offline-прогона."""
    w, h = 640, 480
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        frame = base_frame(w, h)
        x = int((i * 7) % (w - 120))
        cv2.rectangle(frame, (x, 140), (x + 120, 300), (200, 180, 90), -1)
        cv2.circle(frame, (w - x, 360), 46, (90, 180, 200), -1)
        writer.write(frame)
    writer.release()
    return path


def run_offline(video: Path, sessions_dir: Path, conf_path: Path) -> Path:
    """Прогон gesturelab.py --input как в CI/README, настоящим MediaPipe."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONIOENCODING": "utf-8", **__import__("os").environ}
    cmd = [sys.executable, str(REPO / "gesturelab.py"), "--input", str(video),
           "--no-preview", "--out", str(sessions_dir), "--config", str(conf_path),
           "--gesture", "демо", "--person", "Т", "--position", "грудь"]
    print("  $", " ".join(str(c) for c in cmd[1:]))
    res = subprocess.run(cmd, capture_output=True, env=env, cwd=REPO)
    print("  " + (res.stdout.decode("utf-8", "replace").strip() or "(пусто)"))
    if res.returncode != 0:
        print(res.stderr.decode("utf-8", "replace")[-2000:], file=sys.stderr)
        sys.exit(f"offline-прогон завершился кодом {res.returncode}")
    saved = sorted(sessions_dir.glob("*.json"))[-1]
    data = json.loads(saved.read_text(encoding="utf-8"))
    print(f"  сессия {saved.name}: version {data['version']}, kind {data['kind']}, "
          f"summary {json.dumps(data['summary'], ensure_ascii=False)}")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "reports" / "img"),
                    help="папка для PNG (по умолчанию reports/img)")
    ap.add_argument("--image", help="файл кадра: прогнать настоящую детекцию и сохранить PNG")
    ap.add_argument("--video", action="store_true",
                    help="сгенерировать синтетическое видео и прогнать offline-режим")
    args = ap.parse_args()

    out_dir = Path(args.out)
    conf = gesturelab.load_config(REPO / "gesturelab.toml")

    for path in render_hud_states(out_dir):
        print(f"HUD: {path}")

    if args.image:
        frame = imread_u(Path(args.image))
        if frame is None:
            sys.exit(f"Не удалось прочитать {args.image}.")
        print("Детекция на кадре:")
        detect_on_frame(frame, conf, out_dir / f"real_{Path(args.image).stem}.png")

    if args.video:
        tmp = Path(tempfile.mkdtemp(prefix="gl_demo_"))
        video = make_demo_video(tmp / "demo.mp4")
        print(f"Синтетическое видео: {video} ({video.stat().st_size // 1024} КБ)")
        cap = cv2.VideoCapture(str(video))
        ok, first = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Видеозапись {video} не читается: чем её кодировать?")
        print("Детекция на первом кадре видео:")
        detect_on_frame(first, conf, out_dir / "demo_video_frame.png")
        print("Offline-прогон:")
        run_offline(video, tmp / "sessions", REPO / "gesturelab.toml")
        print(f"Артефакты прогона (вне репозитория): {tmp}")


if __name__ == "__main__":
    main()
