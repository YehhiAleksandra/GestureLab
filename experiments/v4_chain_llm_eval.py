# -*- coding: utf-8 -*-
"""LLM-проверка выбранных n-best стратегий на validation-фолдах 3–4."""
import json
import random
import sys
import time
from pathlib import Path

HF = Path(r"D:\hfdata")
ROOT = HF / "GestureLab_PC_v0.1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HF))
from dialogue_state import translate_in_context  # noqa: E402
from gloss_to_text import SYSTEM, Translator, chrf  # noqa: E402
from v4_chain_sweep import FILES, original_rows, predict  # noqa: E402

OUT = HF / "v4_mix" / "v4_chain_llm_validation.json"
PER_SOURCE = 20
BUDGET_S = 250


def main():
    cfg = json.loads((HF / "v4_mix" / "v4_chain_sweep.json").read_text(encoding="utf-8"))
    originals = original_rows()
    rng = random.Random(2026)
    sample = []
    for source, name in FILES.items():
        rows = json.loads((HF / "v4_mix" / name).read_text(encoding="utf-8"))["rows"]
        rows = [r for r in rows if r["fold"] >= 3]
        rng.shuffle(rows)
        selected = cfg["sources"][source]["selected"]
        changed = []
        for r in rows:
            orig = originals[(r["id"], r["phrase"])]
            pred = predict(r, orig, selected["strategy"], selected["es_min"])
            if set(pred) != {w.lower() for w in r["ctc_pred"]}:
                changed.append((source, r, orig, pred))
        sample.extend(changed[:PER_SOURCE])
    previous = {}
    if OUT.exists():
        data = json.loads(OUT.read_text(encoding="utf-8"))
        previous = {r["key"]: r for r in data["rows"]}
    tr = Translator()
    t0 = time.time()
    for i, (source, r, orig, pred) in enumerate(sample):
        key = f"{source}:{r['id']}:{r['phrase']}"
        if key in previous:
            continue
        if time.time() - t0 > BUDGET_S:
            break
        base_glosses = list(zip(orig["pred"], orig["conf"]))
        base = translate_in_context(tr, base_glosses, [], SYSTEM,
                                    chains=orig.get("chains"))
        candidate = translate_in_context(tr, [(w, 0.5) for w in pred], [], SYSTEM)
        previous[key] = {
            "key": key, "source": source, "fold": r["fold"], "id": r["id"],
            "phrase": r["phrase"], "base_pred": orig["pred"], "candidate_pred": pred,
            "base_text": base, "candidate_text": candidate,
            "base_meaning": tr.judge(base, r["phrase"]) if base else 0,
            "candidate_meaning": tr.judge(candidate, r["phrase"]) if candidate else 0,
            "base_chrf": round(chrf(base, r["phrase"]), 3),
            "candidate_chrf": round(chrf(candidate, r["phrase"]), 3),
        }
        OUT.write_text(json.dumps({"rows": list(previous.values())},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(previous)}/{len(sample)} {source} {r['id']}", flush=True)
    summary = {}
    for source in FILES:
        rows = [r for r in previous.values() if r["source"] == source]
        if not rows:
            continue
        summary[source] = {
            "n": len(rows),
            "base_meaning_ge1": round(sum(r["base_meaning"] >= 1 for r in rows) / len(rows), 3),
            "candidate_meaning_ge1": round(sum(r["candidate_meaning"] >= 1 for r in rows) / len(rows), 3),
            "base_chrf": round(sum(r["base_chrf"] for r in rows) / len(rows), 2),
            "candidate_chrf": round(sum(r["candidate_chrf"] for r in rows) / len(rows), 2),
        }
    out = {"method": "changed predictions only; validation folds 3-4; up to 20/source",
           "sample_target": len(sample), "summary": summary,
           "cost_usd": round(tr.cost, 4), "rows": list(previous.values())}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
