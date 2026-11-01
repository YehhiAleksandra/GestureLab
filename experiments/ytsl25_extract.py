# -*- coding: utf-8 -*-
"""VOGinfo: MediaPipe по окнам субтитров → per-cue npz [T,59,3] + мета.

Resume на уровне блоков: <vid>.blocks.json хранит обработанные блоки;
бюджет проверяется внутри видео между блоками.
"""
import json, os, re, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"D:\hfdata\GestureLab_PC_v0.1\tools")
import mediapipe as mp
from dialogue_translator import frame_of

VID = Path(r"D:\hfdata\ytsl25_rsl\videos")
SUBS = Path(r"D:\hfdata\ytsl25_rsl\subs")
CUES = Path(r"D:\hfdata\ytsl25_rsl\cues")
CUES.mkdir(parents=True, exist_ok=True)
PAD = 0.8
MAX_BLOCK_S = 60.0


def parse_vtt(path):
    spans, cur = [], None
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.match(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})", line)
        if m:
            g = [int(x) for x in m.groups()]
            cur = [g[0]*3600+g[1]*60+g[2]+g[3]/1000,
                   g[4]*3600+g[5]*60+g[6]+g[7]/1000, []]
            spans.append(cur)
        elif cur is not None and line.strip() and "-->" not in line:
            cur[2].append(line.strip())
    out = []
    for a, b, tx in spans:
        text = re.sub(r"<[^>]+>", "", " ".join(tx)).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            out.append((a, b, text))
    return out


def blocks_of(cues, dur):
    win = [(max(0, a - PAD), min(dur, b + PAD), i) for i, (a, b, _) in enumerate(cues)]
    win.sort()
    blocks, cur = [], []
    for w in win:
        if cur and (w[0] > cur[-1][1] + 0.1 or w[1] - cur[0][0] > MAX_BLOCK_S):
            blocks.append(cur); cur = []
        cur.append(w)
    if cur:
        blocks.append(cur)
    return blocks


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 260.0
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nshards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    probe = json.load(open(r"D:\hfdata\ytsl25_rsl_probe.json", encoding="utf-8"))
    vids = [v for v, r in probe.items() if r.get("status") == "ok"
            and (VID / f"{v}.mp4").exists() and (SUBS / f"{v}.ru.vtt").exists()]
    if nshards > 1:
        import zlib
        vids = [v for v in vids if zlib.crc32(v.encode()) % nshards == shard]
    # приоритет: недоделанные видео первыми
    vids.sort(key=lambda v: (CUES / f"{v}.blocks.json").exists()
              and json.loads((CUES / f"{v}.blocks.json").read_text(encoding="utf-8")).get("done", False))
    print(f"todo={len(vids)} shard={shard}/{nshards}", flush=True)
    t0 = time.time()
    done_v = 0
    for vid in vids:
        st_path = CUES / f"{vid}.blocks.json"
        st = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() \
            else {"blocks": 0, "meta": [], "done": False}
        if st["done"]:
            continue
        if time.time() - t0 > budget:
            break
        dur = probe.get(vid, {}).get("duration") or 0
        cues = parse_vtt(SUBS / f"{vid}.ru.vtt")
        if not cues:
            st["done"] = True; st_path.write_text(json.dumps(st), encoding="utf-8")
            continue
        blocks = blocks_of(cues, dur if dur else cues[-1][1] + PAD)
        cap = cv2.VideoCapture(str(VID / f"{vid}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        meta = st["meta"]
        with mp.solutions.holistic.Holistic(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=0.3, min_tracking_confidence=0.3) as h:
            for bi in range(st["blocks"], len(blocks)):
                if time.time() - t0 > budget:
                    break
                block = blocks[bi]
                t_start = block[0][0]
                t_end = max(w[1] for w in block)
                cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000)
                frames = []
                t_rel = t_start
                while t_rel < t_end:
                    ok, img = cap.read()
                    if not ok:
                        break
                    frames.append(frame_of(h.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
                    t_rel += 1.0 / fps
                if frames:
                    arr = np.stack(frames).astype(np.float32)
                    base_idx = int(round(t_start * fps))
                    for a, b, ci in block:
                        fa = max(0, int(round((a - t_start) * fps)) - int(0.8 * fps))
                        fb = min(len(arr), int(round((b - t_start) * fps)) + int(0.8 * fps))
                        if fb - fa < 4:
                            continue
                        cid = f"{vid}_{ci:03d}"
                        if any(m["id"] == cid for m in meta):
                            continue
                        np.savez_compressed(CUES / f"{cid}.npz",
                                            frames=arr[fa:fb], fps=fps,
                                            phrase=cues[ci][2],
                                            video=vid, t0=base_idx + fa, t1=base_idx + fb)
                        meta.append({"id": cid, "phrase": cues[ci][2],
                                     "duration": round((fb - fa) / fps, 3)})
                st["blocks"] = bi + 1
                st_path.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        cap.release()
        if st["blocks"] >= len(blocks):
            st["done"] = True
            st_path.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
            done_v += 1
            print(f"  {vid}: блоков {len(blocks)}, cues {len(meta)}, "
                  f"{time.time()-t0:.0f}s", flush=True)
    n_done = sum(1 for v in vids if (CUES / f"{v}.blocks.json").exists()
                 and json.loads((CUES / f"{v}.blocks.json").read_text(encoding="utf-8")).get("done"))
    print(f"session complete_videos={done_v} done_total={n_done}/{len(vids)}", flush=True)


if __name__ == "__main__":
    main()
