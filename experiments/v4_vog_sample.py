# -*- coding: utf-8 -*-
"""Детерминированная held-out выборка ВОГinfo для фьюжн-оценки.

Восстанавливает короткие видеоклипы из исходников по t0/t1 в NPZ и создаёт
уменьшенный progress.json, совместимый с tools/fused_translate.py.
"""
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np

HF = Path(r"D:\hfdata")
MIX = HF / "v4_mix"
OUT_VIDEOS = MIX / "vog_sample_videos"
OUT_WORK = MIX / "work_vog_sample"
PER_FOLD = 40


def main():
    OUT_VIDEOS.mkdir(parents=True, exist_ok=True)
    OUT_WORK.mkdir(parents=True, exist_ok=True)
    progress = json.loads((MIX / "work" / "progress.json").read_text(encoding="utf-8"))
    rng = random.Random(2026)
    sampled = {}
    for fold, payload in sorted(progress["folds"].items(), key=lambda x: int(x[0])):
        rows = [r for r in payload["rows"]
                if re.match(r"^[A-Za-z0-9_-]{11}_\d+$", r["id"])]
        rng.shuffle(rows)
        sampled[fold] = rows[:PER_FOLD]
    made, failed = 0, []
    for fold, rows in sampled.items():
        kept = []
        for r in rows:
            cid = r["id"]
            dst = OUT_VIDEOS / f"{cid}.mp4"
            if not dst.exists():
                npz_path = MIX / "landmarks" / f"{cid}.npz"
                try:
                    with np.load(npz_path, allow_pickle=False) as z:
                        vid = str(z["video"].item())
                        fps = float(z["fps"].item())
                        t0, t1 = int(z["t0"].item()), int(z["t1"].item())
                    cap = cv2.VideoCapture(str(HF / "ytsl25_rsl" / "videos" / f"{vid}.mp4"))
                    if not cap.isOpened():
                        raise RuntimeError("исходное видео не открывается")
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, t0)
                    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("целевой клип не создаётся")
                    for _ in range(max(0, t1 - t0)):
                        ok, frame = cap.read()
                        if not ok:
                            break
                        writer.write(frame)
                    writer.release()
                    cap.release()
                except Exception as exc:
                    failed.append({"id": cid, "error": str(exc)})
                    if dst.exists():
                        dst.unlink()
                    continue
            if dst.exists() and dst.stat().st_size > 1000:
                kept.append(r)
                made += 1
        sampled[fold] = kept
    out = {"meta": dict(progress["meta"], clips=sum(map(len, sampled.values()))),
           "folds": {k: {"rows": v} for k, v in sampled.items()}}
    (OUT_WORK / "progress.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (MIX / "vog_sample_manifest.json").write_text(json.dumps(
        {"seed": 2026, "per_fold": PER_FOLD, "clips": made, "failed": failed},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"clips": made, "failed": len(failed),
                      "videos": str(OUT_VIDEOS), "work": str(OUT_WORK)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
