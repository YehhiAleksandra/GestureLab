#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Открытие источника кадров (камера/видео) и проверка окружения Windows."""

import sys
from pathlib import Path

import cv2


def open_camera(index: int):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        sys.exit("Не удалось открыть камеру. Проверь --camera (попробуй 0, 1, 2).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def open_video(path: str):
    """Различаем «файла нет» и «не открылся кодек»: в Git Bash путь /c/... существует
    для shell, но не для Windows-программы (B26)."""
    if not Path(path).exists():
        sys.exit(f"Видео не найдено: {path}\n"
                 f"В Git Bash передавай путь как C:\\... или D:\\..., а не /c/... или /tmp/...")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Файл {path} есть, но OpenCV его не открыл — проверь кодек/контейнер.")
    return cap


def warn_if_non_ascii_path() -> None:
    """B22: C++-часть MediaPipe открывает графы как ANSI-пути и падает на кириллице."""
    if sys.platform == "win32" and not sys.prefix.isascii():
        print("ВНИМАНИЕ: venv лежит в пути с не-ASCII символами — MediaPipe не загрузит "
              "графы (B22). Создай venv в ASCII-пути, например D:\\venvs\\gesturelab.")
