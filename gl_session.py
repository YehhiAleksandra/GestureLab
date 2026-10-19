#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сессия GestureLab: метаданные, имя файла, полезная нагрузка JSON v2 (читает и v1)."""

import json
import os
import re
import time
from pathlib import Path

SCHEMA_VERSION = 2
KIND_GESTURE = "жест"
KIND_NOISE = "не жест"
POSITIONS = ["грудь", "шнурок", "подставка"]


def pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-а-яА-Я]+", "_", s).strip("_") or "жест"


def kind_of(data: dict) -> str:
    """Миграция v1: без поля kind сессия считается жестом."""
    return data.get("kind", KIND_GESTURE)


def recorded_frames_of(data: dict) -> int:
    """Миграция v1: без recorded_frames считаем записанными все кадры списка."""
    return data["summary"].get("recorded_frames", len(data.get("frames", [])))


def session_filename(session: dict) -> str:
    kind_mark = "" if session["kind"] == KIND_GESTURE else "ne_zhest_"
    return (f"{int(time.time()*1000)}_{kind_mark}"
            f"{safe_name(session['gesture'])}_{session['position']}.json")


def build_payload(session: dict, frames: list, totals: dict) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "kind": session["kind"],
        "gesture": session["gesture"],
        "position": session["position"],
        "person": session["person"],
        "frames": frames,
        "summary": {
            "frames": totals["frames"],
            "recorded_frames": len(frames),
            "both_hands_pct": round(pct(totals["both"], totals["frames"]), 2),
            "face_pct": round(pct(totals["face"], totals["frames"]), 2),
        },
    }


def save_session(out_dir: Path, session: dict, frames: list, totals: dict) -> Path:
    """Атомарная запись с валидацией схемы: битый файл не может появиться в sessions/."""
    import session_schema

    data = build_payload(session, frames, totals)
    errs = session_schema.validate_session(data)
    if errs:
        raise ValueError(f"собранная сессия не проходит схему: {errs}")
    dest = out_dir / session_filename(session)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, dest)
    return dest
