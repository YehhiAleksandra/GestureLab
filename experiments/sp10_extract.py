# -*- coding: utf-8 -*-
"""SP-10 RSL: MediaPipe Holistic -> npz-кэш [T,59,3] + инвентарь phrase-клипов.

Resume по наличию npz. Бюджет времени аргументом (сек).
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, r"D:\hfdata\GestureLab_PC_v0.1")
sys.path.insert(0, r"D:\hfdata\GestureLab_PC_v0.1\tools")

import cv2
import numpy as np

from dialogue_translator import frame_of  # noqa: E402

VID = Path(r"D:\hfdata\sp10_rsl\videos")
CACHE = Path(r"D:\hfdata\sp10_rsl\landmarks")
CACHE.mkdir(parents=True, exist_ok=True)
STATE = Path(r"D:\hfdata\sp10_rsl\download_state.json")

import mediapipe as mp  # noqa: E402


def landmarks(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    with mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.3, min_tracking_confidence=0.3) as h:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(frame_of(h.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
    cap.release()
    return (np.stack(frames).astype(np.float32) if frames else
            np.zeros((0, 59, 3), np.float32)), fps, n


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 260.0
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nshards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    state = json.loads(STATE.read_text(encoding="utf-8"))
    todo = [uid for uid, v in state.items() if v.get("ok")
            and not (CACHE / f"{uid}.npz").exists()]
    if nshards > 1:
        import zlib
        todo = [u for u in todo if (zlib.crc32(u.encode()) % nshards) == shard]
    print(f"todo={len(todo)} shard={shard}/{nshards}", flush=True)
    t0 = time.time()
    done = err = 0
    with mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.3, min_tracking_confidence=0.3) as h:
        for uid in todo:
            if time.time() - t0 > budget:
                break
            path = VID / f"{uid}.mp4"
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames = []
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                frames.append(frame_of(h.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            cap.release()
            arr = (np.stack(frames).astype(np.float32) if frames
                   else np.zeros((0, 59, 3), np.float32))
            np.savez_compressed(CACHE / f"{uid}.npz", frames=arr, fps=fps,
                                phrase=state[uid]["text"])
            done += 1
            if done % 100 == 0:
                print(f"  {done} за {time.time()-t0:.0f}s", flush=True)
    print(f"session: done={done} err={err} rest={len(todo)-done}", flush=True)


if __name__ == "__main__":
    main()
