# -*- coding: utf-8 -*-
"""Подбор использования n-best цепочек + easy_sign на фолдах 0–2, проверка 3–4."""
import json
import sys
from collections import Counter
from pathlib import Path

HF = Path(r"D:\hfdata")
sys.path.insert(0, str(HF / "GestureLab_PC_v0.1" / "tools"))
from eval_phrase_clips import words  # noqa: E402
from rerank_eval import chain_bag, votes_bag  # noqa: E402

FILES = {
    "rutube": "v4_fused_rutube.json",
    "sp10": "v4_fused_sp10.json",
    "voginfo": "v4_fused_vog_sample.json",
}
STRATEGIES = [("top", 1, 1), ("bag", 2, 1), ("bag", 3, 1), ("bag", 5, 1),
              ("votes", 3, 2), ("votes", 5, 2), ("votes", 5, 3)]
THRESHOLDS = [None, 0.5, 0.6, 0.8]


def original_rows():
    p = json.loads((HF / "v4_mix" / "work" / "progress.json").read_text(encoding="utf-8"))
    return {(r["id"], r["phrase"]): r for f in p["folds"].values() for r in f["rows"]}


def predict(row, original, strategy, threshold):
    kind, top, need = strategy
    chains = original.get("chains") or [row["ctc_pred"]]
    if kind == "top":
        bag = chain_bag(chains[:1])
    elif kind == "bag":
        bag = chain_bag(chains[:top])
    else:
        bag = votes_bag(chains[:top], top, need)
    pred = list(bag)
    if threshold is not None:
        for s in sorted((s for s in row["es"] if s["conf"] >= threshold),
                        key=lambda s: s["t"]):
            if s["w"] not in bag:
                pred.append(s["w"])
    return pred


def metrics(items, originals, strategy, threshold):
    hit = pred_n = ref_n = exact = 0
    for r in items:
        pred = predict(r, originals[(r["id"], r["phrase"])], strategy, threshold)
        p, ref = set(pred), set(words(r["phrase"]))
        hit += len(p & ref)
        pred_n += len(p)
        ref_n += len(ref)
        exact += bool(ref) and ref <= p
    recall = hit / max(ref_n, 1)
    precision = hit / max(pred_n, 1)
    return {"clips": len(items), "hits": hit, "pred_unique": pred_n,
            "recall": round(recall, 4), "precision": round(precision, 4),
            "f1": round(2 * recall * precision / max(recall + precision, 1e-12), 4),
            "exact": exact}


def main():
    originals = original_rows()
    out = {"method": "select folds 0-2 by micro F1; validate folds 3-4",
           "sources": {}}
    for source, name in FILES.items():
        rows = json.loads((HF / "v4_mix" / name).read_text(encoding="utf-8"))["rows"]
        tune = [r for r in rows if r["fold"] <= 2]
        val = [r for r in rows if r["fold"] >= 3]
        grid = []
        for strategy in STRATEGIES:
            for threshold in THRESHOLDS:
                grid.append({"strategy": strategy, "es_min": threshold,
                             **metrics(tune, originals, strategy, threshold)})
        best = max(grid, key=lambda r: (r["f1"], r["recall"]))
        strategy, threshold = best["strategy"], best["es_min"]
        result = {"selected": {"strategy": strategy, "es_min": threshold},
                  "tune_best": best,
                  "validation": metrics(val, originals, strategy, threshold),
                  "validation_top1": metrics(val, originals, STRATEGIES[0], None),
                  "validation_top1_es05": metrics(val, originals, STRATEGIES[0], 0.5),
                  "tune_grid": grid}
        out["sources"][source] = result
        print(source, json.dumps(result, ensure_ascii=False))
    dst = HF / "v4_mix" / "v4_chain_sweep.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", dst)


if __name__ == "__main__":
    main()
