#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сводка по сессиям GestureLab (PC и Android — формат одинаковый).

Примеры:
  python summarize.py sessions/                       # сводка по положениям
  python summarize.py sessions/ --by-gesture          # по положениям и жестам
  python summarize.py sessions/ --csv report.csv      # и в CSV
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict


def load_stats(pattern):
    stats = defaultdict(lambda: [0, 0.0, 0.0])  # key -> [frames, both_frames, face_frames]
    files = sorted(glob.glob(os.path.join(pattern, "*.json")))
    if not files:
        sys.exit(f"В {pattern} нет JSON-файлов.")
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            s = d["summary"]
        except Exception as e:
            print(f"  ! пропуск {os.path.basename(f)}: {e}", file=sys.stderr)
            continue
        key = (d.get("position", "?"), d.get("gesture", "?") if BY_GESTURE else None)
        stats[key][0] += s["frames"]
        stats[key][1] += s["both_hands_pct"] / 100.0 * s["frames"]
        stats[key][2] += s["face_pct"] / 100.0 * s["frames"]
    return stats, len(files)


BY_GESTURE = False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="папка с JSON-сессиями")
    ap.add_argument("--by-gesture", action="store_true")
    ap.add_argument("--csv", help="CSV-отчёт")
    args = ap.parse_args()
    BY_GESTURE = args.by_gesture

    stats, n = load_stats(args.folder)

    print(f"Файлов: {n}")
    header = f"{'Положение':12s}" + (f" {'Жест':20s}" if BY_GESTURE else "") +              f" {'Кадров':>7s} {'Обе кисти':>11s} {'Лицо':>8s}  Вердикт"
    print(header)
    print("-" * len(header))
    rows = []
    for (pos, gesture), (n_frames, both, face) in sorted(stats.items()):
        both_pct, face_pct = both / n_frames * 100, face / n_frames * 100
        verdict = "OK >=90%" if both_pct >= 90 else "мимо"
        row = [pos] + ([gesture] if BY_GESTURE else []) +               [n_frames, f"{both_pct:.1f}%", f"{face_pct:.1f}%", verdict]
        rows.append(row)
        print(f"{pos:12s}" + (f" {gesture:20s}" if BY_GESTURE else "") +
              f" {n_frames:7d} {both_pct:10.1f}% {face_pct:7.1f}%  {verdict}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["position"] + (["gesture"] if BY_GESTURE else []) +
                       ["frames", "both_hands_pct", "face_pct", "pass_90"])
            for row in rows:
                w.writerow(row[:-1] + [row[-1] == "OK >=90%"])
        print(f"CSV: {args.csv}")
