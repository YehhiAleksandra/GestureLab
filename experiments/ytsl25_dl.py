# -*- coding: utf-8 -*-
"""Скачивание 697 RSL-видео ВОГinfo (360p mp4) с докачкой."""
import json, os, sys, time

import yt_dlp

VID = r"D:\hfdata\ytsl25_rsl\videos"
os.makedirs(VID, exist_ok=True)
STATE = r"D:\hfdata\ytsl25_rsl\dl_state.json"

probe = json.load(open(r"D:\hfdata\ytsl25_rsl_probe.json", encoding="utf-8"))
state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
todo = [v for v, r in probe.items() if r.get("status") == "ok"]
missing = [v for v in todo if not (os.path.exists(os.path.join(VID, v + ".mp4"))
                                  and os.path.getsize(os.path.join(VID, v + ".mp4")) > 100000)]
print(f"done={len(todo)-len(missing)} missing={len(missing)}")

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 260.0
shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
nshards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
if nshards > 1:
    import zlib
    missing = [v for v in missing if zlib.crc32(v.encode()) % nshards == shard]

t0 = time.time()
ok = fail = 0
for vid in missing:
    if time.time() - t0 > budget:
        break
    try:
        opts = {"format": "134/230/243", "quiet": True, "noprogress": True,
                "outtmpl": os.path.join(VID, vid + ".mp4")}
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([f"https://www.youtube.com/watch?v={vid}"])
        state[vid] = {"ok": True,
                      "size": os.path.getsize(os.path.join(VID, vid + ".mp4"))}
        ok += 1
    except Exception as ex:
        state[vid] = {"ok": False, "err": str(ex)[:140]}
        fail += 1
    if (ok + fail) % 10 == 0:
        json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  {ok+fail} за {time.time()-t0:.0f}s", flush=True)

json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)
print(f"session ok={ok} fail={fail} total={len(state)}")
