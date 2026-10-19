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
