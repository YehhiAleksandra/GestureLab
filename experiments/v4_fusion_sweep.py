# -*- coding: utf-8 -*-
"""Честный подбор порога easy_sign: фолды 0–2 для выбора, 3–4 для проверки."""
import json
import sys
from pathlib import Path

HF = Path(r"D:\hfdata")
ROOT = HF / "GestureLab_PC_v0.1"
sys.path.insert(0, str(ROOT / "tools"))
from eval_phrase_clips import words  # noqa: E402
from fused_translate import merge  # noqa: E402

FILES = {
    "rutube": HF / "v4_mix" / "v4_fused_rutube.json",
    "sp10": HF / "v4_mix" / "v4_fused_sp10.json",
    "voginfo": HF / "v4_mix" / "v4_fused_vog_sample.json",
}
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def metrics(rows, threshold=None):
    hit = pred_n = ref_n = 0
    exact = 0
    for r in rows:
        if threshold is None:
            pred = list(r["ctc_pred"])
        else:
            es = [s for s in r["es"] if float(s["conf"]) >= threshold]
            pred = [w for w, _ in merge(r["ctc_pred"], es, 0.0, 1.0)]
        p, ref = set(pred), set(words(r["phrase"]))
        hit += len(p & ref)
        pred_n += len(p)
        ref_n += len(ref)
        exact += bool(ref) and ref <= p
    recall = hit / max(ref_n, 1)
    precision = hit / max(pred_n, 1)
    return {"clips": len(rows), "hits": hit, "pred_unique": pred_n,
            "ref_unique": ref_n, "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(2 * recall * precision / max(recall + precision, 1e-12), 4),
            "exact": exact}


def main():
    out = {"method": "select folds 0-2 by micro F1; validate folds 3-4",
           "thresholds": THRESHOLDS, "sources": {}}
    for source, path in FILES.items():
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        tune = [r for r in rows if int(r["fold"]) <= 2]
        val = [r for r in rows if int(r["fold"]) >= 3]
        grid = [{"threshold": t, **metrics(tune, t)} for t in THRESHOLDS]
        best = max(grid, key=lambda r: (r["f1"], r["recall"], -r["threshold"]))
        selected = best["threshold"]
        out["sources"][source] = {
            "tune": grid, "selected": selected,
            "validation": {"ctc": metrics(val),
                           "fused": metrics(val, selected),
                           "grid": [{"threshold": t, **metrics(val, t)}
                                    for t in THRESHOLDS]}}
        print(source, "selected", selected,
              "validation", json.dumps(out["sources"][source]["validation"]["fused"],
                                       ensure_ascii=False))
    dst = HF / "v4_mix" / "v4_fusion_sweep.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", dst)


if __name__ == "__main__":
    main()
