import argparse
import json

import numpy as np
import pytest

import gesturelab
import gl_capture
import gl_session
import session_schema


def frame():
    return np.full((240, 320, 3), 40, np.uint8)


class FakeCap:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def run_stub(cap, tmp_path, detectors, conf=None):
    import json
    out = tmp_path / "sessions"
    out.mkdir(exist_ok=True)
    args = argparse.Namespace(input=None, out=str(out), fps=15.0,
                              no_preview_landmarks=True, no_preview=True,
                              gesture="да", person="В", position="грудь", camera=0)
    merged = dict(gesturelab.DEFAULT_CONFIG)
    merged.update(conf or {})
    saved = gesturelab.run_loop(cap, args, out, detectors, False, merged)
    assert cap.released
    return json.loads(saved.read_text(encoding="utf-8"))


def test_broken_frames_are_skipped_not_counted(tmp_path, fake_detectors):
    """Кадр None и кадр нулевого размера не считаются и не роняют цикл записи."""
    cap = FakeCap([frame(), None, np.empty((0, 0, 3), np.uint8), frame()])
    data = run_stub(cap, tmp_path, fake_detectors)
    assert data["summary"]["frames"] == 2


def test_camera_drop_saves_session(tmp_path, fake_detectors, capsys):
    """Обрыв камеры в середине записи сохраняет сессию валидным файлом, а не теряет её."""
    data = run_stub(FakeCap([frame(), frame()]), tmp_path, fake_detectors)
    assert "обрыв камеры" in capsys.readouterr().out
    assert session_schema.validate_session(data) == []


def test_long_session_auto_stops(tmp_path, fake_detectors, capsys):
    """Лимит max_session_seconds предупреждает и сам закрывает сессию: кадры не копятся вечно."""
    cap = FakeCap([frame()] * 6)
    data = run_stub(cap, tmp_path, fake_detectors, {"max_session_seconds": -1.0})
    assert data["summary"]["recorded_frames"] == 1
    assert cap.frames == []                      # источник дочитан, запись больше не шла
    assert len(list((tmp_path / "sessions").glob("*.json"))) == 1
    out = capsys.readouterr().out
    assert "скоро авто-стоп" in out and "авто-стоп по лимиту" in out


def test_save_session_rejects_invalid_payload(tmp_path):
    """Валидация при записи: сессия с расхождением hands/landmarks не попадает на диск."""
    session = {"gesture": "да", "person": "В", "position": "грудь",
               "kind": gl_session.KIND_GESTURE}
    frames = [{"t_ms": 0, "hands": 2, "face": True, "hand_landmarks": [[0.0] * 63]}]
    with pytest.raises(ValueError):
        gl_session.save_session(tmp_path, session, frames,
                                {"frames": 1, "both": 1, "face": 1})
    assert list(tmp_path.iterdir()) == []


def test_fps_of_camera_negative_is_unknown():
    """DSHOW-камера отдаёт -1.0 (B43): отрицательный fps делал все пороги реплики
    отрицательными и ронял живой цикл на пустом буфере."""
    class Cap:
        def __init__(self, v):
            self.v = v

        def get(self, prop):
            return self.v

    assert gl_capture.fps_of(Cap(-1.0)) == 25.0
    assert gl_capture.fps_of(Cap(0.0)) == 25.0
    assert gl_capture.fps_of(Cap(30.0)) == 30.0


def test_open_video_missing_file_hints_at_path_style(tmp_path):
    """Отсутствующее видео — отдельное сообщение с подсказкой про путь MSYS, а не «кодек» (B26)."""
    with pytest.raises(SystemExit) as exc:
        gl_capture.open_video(str(tmp_path / "нет.mp4"))
    msg = str(exc.value)
    assert "не найдено" in msg and "Git Bash" in msg


def test_open_video_undecodable_file_is_not_missing(tmp_path):
    """Файл есть, но не видео: ошибка про кодек, а не «не найдено»."""
    fake = tmp_path / "not_video.mp4"
    fake.write_text("не mp4", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        gl_capture.open_video(str(fake))
    msg = str(exc.value)
    assert "не открыл" in msg and "не найдено" not in msg
