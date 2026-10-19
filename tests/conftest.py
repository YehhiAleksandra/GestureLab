import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture(scope="session")
def fixtures_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def golden_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "golden"


@pytest.fixture(scope="session")
def repo_python() -> str:
    return sys.executable


@pytest.fixture(scope="session")
def utf8_env() -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class _Pt:
    __slots__ = ("x", "y", "z")

    def __init__(self, v):
        self.x = self.y = v
        self.z = 0.0


class _Hand:
    def __init__(self):
        self.landmark = [_Pt(i / 63.0) for i in range(21)]


class _ResH:
    multi_hand_landmarks = [_Hand(), _Hand()]


class _ResF:
    detections = [object()]


class FakeHands:
    def process(self, rgb):
        return _ResH()


class FakeFace:
    def process(self, rgb):
        return _ResF()


@pytest.fixture
def fake_detectors():
    """Подставные детекторы: всегда две кисти и лицо, без mediapipe."""
    return FakeHands(), FakeFace()
