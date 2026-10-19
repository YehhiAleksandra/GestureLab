#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Схема JSON-сессии GestureLab (v1 и v2): проверка структуры без тяжёлых зависимостей."""

import json

SCHEMA_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
REQUIRED_TOP = ("version", "gesture", "position", "person", "frames", "summary")
REQUIRED_FRAME = ("t_ms", "hands", "face", "hand_landmarks")
REQUIRED_SUMMARY = ("frames", "both_hands_pct", "face_pct")
LANDMARK_DIM = 63  # 21 точка кисти × 3 координаты


def validate_session(data) -> list:
    """Возвращает список ошибок схемы; пустой список = сессия валидна."""
    if not isinstance(data, dict):
        return ["корень JSON — не объект"]
    errs = [f"нет поля {k!r}" for k in REQUIRED_TOP if k not in data]
    if errs:
        return errs
    if data["version"] not in SUPPORTED_VERSIONS:
        errs.append(f"version={data['version']!r}, поддерживаются {SUPPORTED_VERSIONS}")
    if data["version"] >= 2 and "kind" not in data:
        errs.append("нет поля 'kind' (обязательно с version 2)")

    frames = data["frames"]
    if not isinstance(frames, list):
        errs.append("frames — не список")
        frames = []
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            errs.append(f"frames[{i}] — не объект")
            continue
        errs += [f"frames[{i}]: нет поля {k!r}" for k in REQUIRED_FRAME if k not in fr]
        lms = fr.get("hand_landmarks")
        if not isinstance(lms, list) or len(lms) != fr.get("hands", 0):
            errs.append(f"frames[{i}]: hand_landmarks не соответствует hands")
        elif any(not isinstance(v, list) or len(v) != LANDMARK_DIM for v in lms):
            errs.append(f"frames[{i}]: вектор точек не {LANDMARK_DIM} чисел")

    s = data["summary"]
    if not isinstance(s, dict):
        errs.append("summary — не объект")
    else:
        errs += [f"summary: нет поля {k!r}" for k in REQUIRED_SUMMARY if k not in s]
    return errs


def read_session(path):
    """(data или None, список ошибок): нечитаемый файл — тоже ошибка схемы."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        return None, [f"файл не читается как JSON: {e}"]
    return data, validate_session(data)
