# -*- coding: utf-8 -*-
"""Скачивание ручных русских субтитров для RSL-видео YouTube-SL-25 (VOGinfo)."""
import json, os, sys, time

import yt_dlp

PROBE = r"D:\hfdata\ytsl25_rsl_probe.json"
OUT = r"D:\hfdata\ytsl25_rsl\subs"
os.makedirs(OUT, exist_ok=True)
STATE = r"D:\hfdata\ytsl25_subs_state.json"

state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
probe = json.load(open(PROBE, encoding="utf-8"))
todo = [v for v, r in probe.items()
        if r.get("status") == "ok" and v not in state]

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
t0 = time.time()
done = fail = 0
ydl_opts = {
    "skip_download": True,
    "writesubtitles": True,
    "subtitleslangs": ["ru"],
    "subtitlesformat": "vtt",
    "quiet": True,
    "no_warnings": True,
    "noprogress": True,
    "outtmpl": os.path.join(OUT, "%(id)s.%(ext)s"),
}
for vid in todo:
    if time.time() - t0 > budget:
        break
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as y:
            y.download([f"https://www.youtube.com/watch?v={vid}"])
        state[vid] = {"ok": os.path.exists(os.path.join(OUT, vid + ".ru.vtt"))}
    except Exception as ex:
        state[vid] = {"ok": False, "err": str(ex)[:120]}
    done += 1
    if done % 25 == 0:
        json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  {done} за {time.time()-t0:.0f}s", flush=True)

json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)
ok = sum(1 for v in state.values() if v.get("ok"))
print(f"session done={done} state_ok={ok}/{len(state)}")
