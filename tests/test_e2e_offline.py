import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import gesturelab
import session_schema

REPO = Path(__file__).resolve().parent.parent


def make_video(path: Path, n_frames=60, size=(320, 240)):
    """Синтетический ролик: две движущиеся «кисти» на тёмном фоне."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert w.isOpened(), f"VideoWriter не открылся для {path}"
    for i in range(n_frames):
        f = np.full((size[1], size[0], 3), 40, np.uint8)
        x = 40 + (i * 4) % (size[0] - 80)
        cv2.circle(f, (x, 120), 30, (220, 190, 170), -1)
        cv2.circle(f, (size[0] - x, 120), 30, (220, 190, 170), -1)
        w.write(f)
    w.release()


def stub_run(tmp_path, fake_detectors, n_frames=40):
    vid = tmp_path / "stub.mp4"
    make_video(vid, n_frames=n_frames)
    out = tmp_path / "s"
    out.mkdir()
    cap = cv2.VideoCapture(str(vid))
    args = argparse.Namespace(input=str(vid), out=str(out), fps=15.0,
                              no_preview_landmarks=True, no_preview=True,
                              gesture="тест", person="Т", position="грудь", camera=0)
    saved = gesturelab.run_loop(cap, args, out, fake_detectors, False,
                                dict(gesturelab.DEFAULT_CONFIG))
    return json.loads(saved.read_text(encoding="utf-8"))


def test_offline_video_produces_valid_session(tmp_path, repo_python, utf8_env):
    """Offline-прогон видеофайла пишет одну валидную сессию с процентами в [0, 100]."""
    vid = tmp_path / "hands.mp4"
    make_video(vid)
    out = tmp_path / "sessions"
    r = subprocess.run(
        [repo_python, str(REPO / "gesturelab.py"), "--input", str(vid),
         "--no-preview", "--out", str(out), "--gesture", "тест", "--person", "Т"],
        capture_output=True, env=utf8_env, timeout=300)
    if r.returncode != 0 and b"path does not exist" in r.stderr and not sys.prefix.isascii():
        pytest.skip("B22: MediaPipe не грузит графы из не-ASCII пути; venv нужен вне кириллицы")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    files = list(out.glob("*.json"))
    assert len(files) == 1
    data, errs = session_schema.read_session(files[0])
    assert errs == []
    assert data["summary"]["frames"] > 0
    assert data["gesture"] == "тест" and data["person"] == "Т"
    for key in ("both_hands_pct", "face_pct"):
        assert 0.0 <= data["summary"][key] <= 100.0


def test_stubbed_detector_session_is_full_percent(tmp_path, fake_detectors):
    """На подставном детекторе сессия даёт 100% обеих кистей и лица, векторы по 63."""
    data = stub_run(tmp_path, fake_detectors)
    assert data["summary"]["both_hands_pct"] == 100.0
    assert data["summary"]["face_pct"] == 100.0
    assert len(data["frames"]) >= 1
    assert all(len(h) == 63 for h in data["frames"][0]["hand_landmarks"])
    t = [fr["t_ms"] for fr in data["frames"]]
    assert t == sorted(t)


def test_recorded_frames_counted_separately(tmp_path, fake_detectors):
    """B3 закрыт: summary.frames — кадры камеры, recorded_frames — сколько записано."""
    data = stub_run(tmp_path, fake_detectors)
    assert data["version"] == 2 and data["kind"] == "жест"
    recorded = data["summary"]["recorded_frames"]
    assert recorded == len(data["frames"])
    assert recorded <= data["summary"]["frames"]
