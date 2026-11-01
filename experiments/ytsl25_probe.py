# -*- coding: utf-8 -*-
"""Проба: 715 RSL-видео YouTube-SL-25 — доступность и русские субтитры."""
import csv, json, sys, time

import yt_dlp

CSV = r"D:\ДляГлухихAi\GestureLab_PC_v0.1\datasets\signverse_test\SignVerse-2M-metadata_split.csv"
OUT = r"D:\hfdata\ytsl25_rsl_probe.json"

rows = [r for r in csv.reader(open(CSV, encoding="utf-8")) if len(r) == 3 and r[1] == "rsl"]
rows = rows[1:] if rows and rows[0][0] == "video_id" else rows
print("rsl rows:", len(rows))

state = json.load(open(OUT, encoding="utf-8")) if __import__("os").path.exists(OUT) else {}

ydl_opts = {
    "skip_download": True,
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["ru", "ru-RU"],
    "quiet": True,
    "no_warnings": True,
}

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
t0 = time.time()
n = 0
for vid, lang, split in rows:
    if vid in state or time.time() - t0 > budget:
        continue
    rec = {"split": split}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as y:
            info = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
        if info is None:
            rec["status"] = "gone"
        else:
            rec["status"] = "ok"
            rec["duration"] = info.get("duration")
            rec["title"] = (info.get("title") or "")[:100]
            rec["channel"] = (info.get("channel") or "")[:60]
            subs = info.get("subtitles") or {}
            auto = info.get("automatic_captions") or {}
            rec["subs_ru"] = sorted(subs.keys())[:5]
            rec["auto_ru"] = [l for l in auto.keys() if l.startswith("ru")][:5]
    except Exception as ex:
        rec["status"] = "err"
        rec["err"] = str(ex)[:140]
    state[vid] = rec
    n += 1
    if n % 10 == 0:
        json.dump(state, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  {n} probed, {time.time()-t0:.0f}s", flush=True)

json.dump(state, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
import collections
c = collections.Counter(v.get("status") for v in state.values())
withsub = sum(1 for v in state.values() if v.get("subs_ru") or v.get("auto_ru"))
print("probed:", len(state), dict(c), "with_ru_subs:", withsub)
