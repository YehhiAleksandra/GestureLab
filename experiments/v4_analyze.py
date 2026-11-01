# -*- coding: utf-8 -*-
"""Разбор CV-метрик v4 по источникам: rutube / SP-10 / ВОГinfo.

Читает progress.json (работает и на частичных фолдах) и считает word_recall
и phrase_all_words для каждого источника отдельно — это сравнение с v3
на тех же самых клипах, а не среднее по разнородному корпусу.
"""
import json
import re
import sys
from collections import defaultdict

HF = r"D:/hfdata"
PROG = HF + r"/v4_mix/work/progress.json"
RUTUBE = json.load(open(HF + r"/combined_phrases/rutube_ids.json", encoding="utf-8"))
RUTUBE = set(RUTUBE)


def source(cid):
    if cid in RUTUBE:
        return "rutube"
    if re.match(r"^[A-Za-z0-9_-]{11}_\d+$", cid):
        return "voginfo"
    return "sp10"


def summarize(rows):
    agg = defaultdict(lambda: [0, 0, 0, 0])  # hits, total, exact, n
    for r in rows:
        s = source(r["id"])
        a = agg[s]
        a[0] += int(r["hits"])
        a[1] += int(r["total"])
        a[2] += 1 if int(r["total"]) and int(r["hits"]) == int(r["total"]) else 0
        a[3] += 1
    return agg


def main(path):
    p = json.load(open(path, encoding="utf-8"))
    all_agg = defaultdict(lambda: [0, 0, 0, 0])
    for k, f in sorted(p.get("folds", {}).items(), key=lambda x: int(x[0])):
        agg = summarize(f["rows"])
        print(f"== фолд {k} (held {len(f['rows'])}) ==")
        for s in ("rutube", "sp10", "voginfo"):
            hits, total, exact, n = agg[s]
            if not n:
                continue
            wr = hits / total if total else 0.0
            print(f"  {s:8s} n={n:5d} word_recall={wr:.3f} "
                  f"phrase_all_words={exact}/{n} ({exact/n:.3f})")
            for i in range(4):
                all_agg[s][i] += agg[s][i]
    print("== ИТОГО по готовым фолдам ==")
    for s in ("rutube", "sp10", "voginfo"):
        hits, total, exact, n = all_agg[s]
        if not n:
            continue
        wr = hits / total if total else 0.0
        print(f"  {s:8s} n={n:5d} word_recall={wr:.3f} "
              f"phrase_all_words={exact}/{n} ({exact/n:.3f})")
    print("v3 для справки: rutube held 0.179, SP-10 held 0.106")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PROG)
