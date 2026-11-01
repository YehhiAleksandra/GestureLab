# -*- coding: utf-8 -*-
"""VOGinfo: слить *.blocks.json → инвентарь phrase-клипов (cue-level).

Выход: D:/hfdata/ytsl25_rsl/ytsl25_inventory.json совместим с load_phrase_clips
(finetune_ctc_phrases.py): cache = D:/hfdata/ytsl25_rsl/cues, один npz на id.
"""
import json
from pathlib import Path

CUES = Path(r"D:\hfdata\ytsl25_rsl\cues")
OUT = Path(r"D:\hfdata\ytsl25_rsl\ytsl25_inventory.json")

phrases = []
for f in sorted(CUES.glob("*.blocks.json")):
    try:
        st = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if not st.get("done"):
        continue
    phrases.extend(st.get("meta", []))

inv = {"channel": "VOGinfo / YouTube-SL-25 RSL (cue-level, ru subtitles)",
       "date": "2026-10-29",
       "counts": {"phrases": len(phrases)},
       "phrases": phrases, "long": []}
OUT.write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"cues: {len(phrases)} -> {OUT}")
if phrases:
    lens = sorted(p["duration"] for p in phrases if p.get("duration"))
    print("dur median:", lens[len(lens)//2], "p95:", lens[int(len(lens)*0.95)])
