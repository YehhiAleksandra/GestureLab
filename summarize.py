#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сводка по сессиям GestureLab (PC и Android — формат одинаковый).
Сессии «не жест» по умолчанию не учитываются (это фон этапа 1, не метрика этапа 0).

Примеры:
  python summarize.py sessions/                       # сводка по положениям
  python summarize.py sessions/ --by-gesture          # по положениям и жестам
  python summarize.py sessions/ --include-noise       # вместе с сессиями «не жест»
  python summarize.py sessions/ --csv report.csv      # и в CSV
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

from gl_session import KIND_NOISE, kind_of


def load_sessions(pattern, include_noise=False):
    """(валидные сессии, число найденных файлов, пропуски[(имя, причина)])."""
    files = sorted(glob.glob(os.path.join(pattern, "*.json")))
    good, skipped = [], []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            s = d["summary"]
            s["frames"], s["both_hands_pct"], s["face_pct"]
        except Exception as e:
            skipped.append((os.path.basename(f), str(e)))
            continue
        if not include_noise and kind_of(d) == KIND_NOISE:
            skipped.append((os.path.basename(f), "сессия «не жест»"))
            continue
        good.append(d)
    return good, len(files), skipped


def aggregate(sessions, by_gesture=False):
    """key (position[, gesture]) -> [кадров, взвешенные обе кисти, взвешенное лицо]."""
    stats = defaultdict(lambda: [0, 0.0, 0.0])
    for d in sessions:
        s = d["summary"]
        key = (d.get("position", "?"), d.get("gesture", "?") if by_gesture else None)
        stats[key][0] += s["frames"]
        stats[key][1] += s["both_hands_pct"] / 100.0 * s["frames"]
        stats[key][2] += s["face_pct"] / 100.0 * s["frames"]
    return stats


def render(stats, by_gesture):
    header = (f"{'Положение':12s}" + (f" {'Жест':20s}" if by_gesture else "") +
              f" {'Кадров':>7s} {'Обе кисти':>11s} {'Лицо':>8s}  Вердикт")
    lines = [header, "-" * len(header)]
    rows = []
    for (pos, gesture), (n_frames, both, face) in sorted(stats.items()):
        both_pct = both / n_frames * 100 if n_frames else 0.0
        face_pct = face / n_frames * 100 if n_frames else 0.0
        verdict = "OK >=90%" if both_pct >= 90 else ("нет кадров" if not n_frames else "мимо")
        row = ([pos] + ([gesture] if by_gesture else []) +
               [n_frames, f"{both_pct:.1f}%", f"{face_pct:.1f}%", verdict])
        rows.append(row)
        lines.append(f"{pos:12s}" + (f" {gesture:20s}" if by_gesture else "") +
                     f" {n_frames:7d} {both_pct:10.1f}% {face_pct:7.1f}%  {verdict}")
    return lines, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="папка с JSON-сессиями")
    ap.add_argument("--by-gesture", action="store_true")
    ap.add_argument("--include-noise", action="store_true",
                    help="учитывать сессии «не жест»")
    ap.add_argument("--csv", help="CSV-отчёт")
    args = ap.parse_args()

    sessions, n_found, skipped = load_sessions(args.folder, args.include_noise)
    if not n_found:
        sys.exit(f"В {args.folder} нет JSON-файлов.")
    for name, why in skipped:
        print(f"  ! пропуск {name}: {why}", file=sys.stderr)
    print(f"Файлов: {n_found} (разобрано {len(sessions)})")

    lines, rows = render(aggregate(sessions, args.by_gesture), args.by_gesture)
    print("\n".join(lines))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["position"] + (["gesture"] if args.by_gesture else []) +
                       ["frames", "both_hands_pct", "face_pct", "pass_90"])
            for row in rows:
                w.writerow(row[:-1] + [row[-1] == "OK >=90%"])
        print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
