#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пересборка золотых файлов сводки из tests/fixtures (детерминированно).

Пример: python tools/regen_golden.py
Запускать после намеренного изменения вывода summarize.py; затем git diff
tests/golden — в диффе должны быть только ожидаемые строки.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"
GOLDEN = REPO / "tests" / "golden"


def capture(*argv) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run([sys.executable, str(REPO / "summarize.py"), str(FIX), *argv],
                         capture_output=True, env=env, cwd=REPO)
    if res.returncode != 0:
        sys.exit(res.stderr.decode("utf-8", "replace"))
    return res.stdout.decode("utf-8", "replace").replace("\r\n", "\n")


def main() -> None:
    for name, argv in (("summarize_fixtures.txt", []),
                       ("summarize_fixtures_by_gesture.txt", ["--by-gesture"])):
        (GOLDEN / name).write_text(capture(*argv).rstrip() + "\n",
                                   encoding="utf-8", newline="\n")
        print(f"{name} ok")


if __name__ == "__main__":
    main()
