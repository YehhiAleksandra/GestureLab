# -*- coding: utf-8 -*-
"""Связь качества cue-фрагментов ВОГinfo с held-out распознаванием v4."""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, r"D:\hfdata\GestureLab_PC_v0.1")
from sign_utils import split_utterances
sys.path.insert(0, r"D:\hfdata\GestureLab_PC_v0.1\tools")
from eval_phrase_clips import words

HF = Path(r"D:\hfdata")
CACHE = HF / "v4_mix" / "landmarks"
OUT = HF / "v4_mix" / "v4_quality_analysis.json"


def safe(v):
    return float(v) if np.isfinite(v) else 0.0


def metrics(row):
    path = CACHE / (row["id"] + ".npz")
    if not path.exists():
        return None
    with np.load(path) as z:
        x = z["frames"]
        fps = float(z["fps"])
    if not len(x):
        return None
    hands = x[:, :42].reshape(len(x), 2, 21, 3)
    present = ~np.isnan(hands[:, :, 0, 0]).all(1)
    both = ~np.isnan(hands[:, :, 0, 0]).any(1)
    wrists = hands[:, :, 0, :2]
    delta = np.abs(np.diff(wrists, axis=0))
    motion = np.nanmean(delta) if len(delta) else 0.0
    spans = split_utterances(x, fps, pause=0.7, min_len=0.4, max_len=15.0)
    a, b = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(x))
    nw = len(words(row["phrase"]))
    dur = len(x) / fps
    return {"hand": safe(present.mean()), "both": safe(both.mean()),
            "motion": safe(motion), "used": (b - a) / len(x),
            "wps": nw / dur if dur else 0.0, "words": nw, "duration": dur,
            "frames": len(x)}


def summarize(rows, key, bins):
    groups = defaultdict(list)
    for r in rows:
        v = r[key]
        for lo, hi, name in bins:
            if lo <= v < hi:
                groups[name].append(r)
                break
    out = {}
    for name, rs in groups.items():
        hits = sum(int(r["hits"]) for r in rs)
        total = sum(int(r["total"]) for r in rs)
        out[name] = {"clips": len(rs), "recall": round(hits / total, 4) if total else 0,
                     "nonempty_hits": round(sum(r["hits"] > 0 for r in rs) / len(rs), 4),
                     "tokens": total}
    return out


def main():
    rep = json.loads((HF / "v4_mix" / "v4_cv.json").read_text(encoding="utf-8"))
    rows = [r for r in rep["rows"] if re.match(r"^[A-Za-z0-9_-]{11}_\d+$", r["id"])]
    kept = []
    for i, row in enumerate(rows):
        m = metrics(row)
        if m:
            kept.append(dict(row, **m))
        if (i + 1) % 1000 == 0:
            print(f"{i+1}/{len(rows)}", flush=True)
    result = {"clips": len(kept), "method": "VOGinfo held-out v4; token recall",
              "overall": {"hits": sum(r["hits"] for r in kept),
                          "tokens": sum(r["total"] for r in kept),
                          "recall": round(sum(r["hits"] for r in kept) /
                                          sum(r["total"] for r in kept), 4)},
              "by_hand": summarize(kept, "hand", [(0, .5, "<0.5"), (.5, .8, "0.5-0.8"),
                                                   (.8, .95, "0.8-0.95"), (.95, 1.01, ">=0.95")]),
              "by_used": summarize(kept, "used", [(0, .5, "<0.5"), (.5, .8, "0.5-0.8"),
                                                   (.8, .99, "0.8-0.99"), (.99, 1.01, ">=0.99")]),
              "by_wps": summarize(kept, "wps", [(0, 1, "<1"), (1, 2, "1-2"), (2, 3, "2-3"),
                                                 (3, 4, "3-4"), (4, 100, ">=4")]),
              "by_motion": summarize(kept, "motion", [(0, .0005, "<0.0005"),
                                                       (.0005, .002, "0.0005-0.002"),
                                                       (.002, .01, "0.002-0.01"),
                                                       (.01, 100, ">=0.01")]),
              "rows": kept}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
