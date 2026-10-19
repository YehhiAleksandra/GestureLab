#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор фиктур JSON-сессий GestureLab (детерминированный, без mediapipe).

Пример: python make_fixture.py
Кладёт 8 файлов в tests/fixtures/, включая граничные: 0 кадров, нет флага face,
битый JSON, сессию «не жест» (схема v2). Перезапуск даёт байт-в-байт тот же
результат (золотые файлы сводки).
"""

import json
from pathlib import Path

FIX = Path(__file__).parent / "tests" / "fixtures"


def lm(seed: int) -> list:
    return [round(((seed * 7 + i) % 100) / 100.0, 4) for i in range(63)]


def session(gesture, position, person, n_frames, hands, face):
    frames = [{"t_ms": i * 67, "hands": hands, "face": face,
               "hand_landmarks": [lm(i + 1) for _ in range(hands)]}
              for i in range(n_frames)]
    return {
        "version": 1,
        "gesture": gesture,
        "position": position,
        "person": person,
        "frames": frames,
        "summary": {
            "frames": n_frames,
            "both_hands_pct": round(100.0 if hands >= 2 else 0.0, 2),
            "face_pct": round(100.0 if face else 0.0, 2),
        },
    }


def dump(name: str, data) -> None:
    (FIX / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    dump("fix_both_hands.json", session("да", "грудь", "В", 30, 2, True))
    dump("fix_one_hand.json", session("нет", "подставка", "В", 20, 1, False))
    dump("fix_face_only.json", session("вопрос", "грудь", "А", 25, 0, True))
    dump("fix_both_hands_shnurok.json", session("да", "шнурок", "А", 40, 2, True))
    dump("fix_zero_frames.json", session("да", "грудь", "В", 0, 2, True)
         | {"frames": [], "summary": {"frames": 0, "both_hands_pct": 0.0, "face_pct": 0.0}})
    no_face = session("да", "шнурок", "А", 10, 2, True)
    for fr in no_face["frames"]:
        del fr["face"]
    dump("fix_no_face_flag.json", no_face)
    noise = session("телефон", "грудь", "В", 15, 2, True)
    noise["version"] = 2
    noise["kind"] = "не жест"
    noise["summary"]["recorded_frames"] = len(noise["frames"])
    dump("fix_noise.json", noise)
    (FIX / "fix_broken.json").write_text("{ not json", encoding="utf-8")
    print(f"фикстуры: {FIX} — {len(list(FIX.glob('*.json')))} файлов")


if __name__ == "__main__":
    main()
