# -*- coding: utf-8 -*-
"""LLM-оценка смысла v4 на стратифицированной выборке held-строк CV.

Выборка: 20 rutube + 20 sp10 + 40 voginfo (детерминированная).
Инкрементально: результат каждой строки пишется в out JSON, повторный запуск
докачивает недостающее (API бывает нестабильным).
"""
import json
import random
import re
import sys
import time
from pathlib import Path

HF = Path(r"D:/hfdata")
sys.path.insert(0, str(HF / "GestureLab_PC_v0.1"))
sys.path.insert(0, str(HF / "GestureLab_PC_v0.1" / "tools"))
from dialogue_state import translate_in_context  # noqa: E402
from gloss_to_text import SYSTEM, Translator, chrf  # noqa: E402

OUT = HF / "v4_mix" / "v4_llm_sample.json"
RUTUBE = set(json.load(open(HF / "combined_phrases" / "rutube_ids.json",
                            encoding="utf-8")))
QUOTA = {"rutube": 20, "sp10": 20, "voginfo": 40}


def source(cid):
    if cid in RUTUBE:
        return "rutube"
    if re.match(r"^[A-Za-z0-9_-]{11}_\d+$", cid):
        return "voginfo"
    return "sp10"


def main():
    rows = json.load(open(HF / "v4_mix" / "v4_cv.json", encoding="utf-8"))["rows"]
    rng = random.Random(2026)
    by_src = {"rutube": [], "sp10": [], "voginfo": []}
    for r in rows:
        by_src[source(r["id"])].append(r)
    sample = []
    for s, q in QUOTA.items():
        pool = by_src[s]
        rng.shuffle(pool)
        sample.extend(pool[:q])
    prev = {}
    if OUT.exists():
        for r in json.load(open(OUT, encoding="utf-8")):
            prev[r["id"]] = r
    done = 0
    t0 = time.time()
    tr = Translator()
    for r in sample:
        if r["id"] in prev:
            done += 1
            continue
        if time.time() - t0 > 250:  # бюджет на прогон, остальное — докачкой
            break
        glosses = list(zip(r["pred"], r["conf"]))
        text = translate_in_context(tr, glosses, [], SYSTEM,
                                    chains=r.get("chains"))
        rec = dict(r)
        rec["translation"] = text
        rec["meaning"] = tr.judge(text, r["phrase"]) if text else 0
        rec["chrf"] = round(chrf(text, r["phrase"]), 3)
        prev[r["id"]] = rec
        done += 1
        print(f"{done}/{len(sample)} {r['id']} meaning={rec['meaning']} "
              f"chrf={rec['chrf']}", flush=True)
        OUT.write_text(json.dumps(list(prev.values()), ensure_ascii=False, indent=1),
                       encoding="utf-8")
    judged = [r for r in prev.values() if r.get("meaning") is not None]
    for s in ("rutube", "sp10", "voginfo"):
        sub = [r for r in judged if source(r["id"]) == s]
        if not sub:
            continue
        ge1 = sum(r["meaning"] >= 1 for r in sub) / len(sub)
        ch = sum(r["chrf"] for r in sub) / len(sub)
        print(f"{s}: n={len(sub)} meaning_ge1={ge1:.3f} chrf={ch:.1f}")
    print("готово:", len(judged), "/", len(sample))


if __name__ == "__main__":
    main()
