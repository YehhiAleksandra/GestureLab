# -*- coding: utf-8 -*-
"""Быстрый sweep декодера v4: один forward на клип, resume после каждого клипа.

Выборка: все 67 rutube + 300 SP-10 + 200 ВОГinfo. Параметры выбираются на
фолдах 0–2 и проверяются на 3–4. LM выключена (лучший режим прошлых этапов).
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HF = Path(r"D:\hfdata")
ROOT = HF / "GestureLab_PC_v0.1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from dialogue_translator import Recognizer  # noqa: E402
from eval_phrase_clips import hits  # noqa: E402
from finetune_ctc_phrases import load_phrase_clips  # noqa: E402
from sign_beam import ctc_beam_search, smooth_log_probs  # noqa: E402
from sign_utils import features  # noqa: E402

OUT = HF / "v4_mix" / "decode_sweep"
CONFIGS = [{"word_bonus": b, "smooth": s, "beam": 24}
           for b in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0) for s in (1, 3)]


def selected_rows():
    result = []
    for source, name, limit in (("rutube", "v4_fused_rutube.json", 67),
                                ("sp10", "v4_fused_sp10.json", 300),
                                ("voginfo", "v4_fused_vog_sample.json", 200)):
        rows = json.loads((HF / "v4_mix" / name).read_text(encoding="utf-8"))["rows"]
        rng = random.Random(2026 + len(source))
        rng.shuffle(rows)
        if source == "sp10":
            rows = sorted(rows, key=lambda r: (r["fold"], r["id"], r["phrase"]))
            by_fold = defaultdict(list)
            for r in rows:
                by_fold[r["fold"]].append(r)
            rows = [r for k in sorted(by_fold) for r in by_fold[k][:60]]
        for r in rows[:limit] if source != "sp10" else rows:
            result.append({"uid": f"{source}:{len(result)}:{r['id']}", "source": source,
                           "fold": int(r["fold"]), "id": r["id"], "phrase": r["phrase"]})
    return result


def summarize(rows):
    hit = sum(hits(r["pred"], r["phrase"])[0] for r in rows)
    pred_n = sum(len(r["pred"]) for r in rows)
    ref_n = sum(len(r["phrase"].lower().replace("ё", "е").split()) for r in rows)
    # words() через hits с пустым предсказанием не даёт число токенов; используем его корректно.
    from eval_phrase_clips import words
    ref_n = sum(len(words(r["phrase"])) for r in rows)
    recall = hit / max(ref_n, 1)
    precision = hit / max(pred_n, 1)
    return {"clips": len(rows), "hits": hit, "pred_tokens": pred_n,
            "ref_tokens": ref_n, "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(2 * recall * precision / max(recall + precision, 1e-12), 4)}


def aggregate(selection):
    all_rows = []
    for k in range(5):
        p = OUT / f"fold{k}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        if len(data["rows"]) < len(CONFIGS) * sum(s["fold"] == k for s in selection):
            return None
        all_rows.extend(data["rows"])
    by_uid = {s["uid"]: s for s in selection}
    report = {"method": "tune folds 0-2, validate folds 3-4; token micro F1",
              "configs": CONFIGS, "sources": {}}
    for source in ("all", "rutube", "sp10", "voginfo"):
        tune, val = [], []
        for ci, cfg in enumerate(CONFIGS):
            rows = [r for r in all_rows if r["config"] == ci and
                    (source == "all" or by_uid[r["uid"]]["source"] == source)]
            tr = [r for r in rows if by_uid[r["uid"]]["fold"] <= 2]
            va = [r for r in rows if by_uid[r["uid"]]["fold"] >= 3]
            tune.append({"config": ci, **cfg, **summarize(tr)})
            val.append({"config": ci, **cfg, **summarize(va)})
        best = max(tune, key=lambda r: (r["f1"], r["recall"]))
        chosen = next(r for r in val if r["config"] == best["config"])
        baseline = next(r for r in val if r["config"] == 6)  # bonus 3, smooth 1
        report["sources"][source] = {"selected": best["config"], "tune_best": best,
                                     "validation": chosen, "validation_baseline": baseline,
                                     "tune_grid": tune, "validation_grid": val}
        print(source, "selected", best["config"], json.dumps(chosen, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = selected_rows()
    done = aggregate(selection)
    if done:
        print("все фолды уже готовы")
        return
    ck = torch.load(HF / "v4_mix" / "work" / "fold0.pt", map_location="cpu",
                    weights_only=False)
    clips, skipped = load_phrase_clips(str(HF / "v4_mix" / "landmarks"),
                                      str(HF / "v4_mix" / "v4_mix_inventory.json"),
                                      list(ck["names"]), {})
    clip_map = {(c["id"], c["phrase"]): c for c in clips}
    for fold in range(5):
        items = [s for s in selection if s["fold"] == fold]
        path = OUT / f"fold{fold}.json"
        data = {"rows": []}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        done_keys = {(r["uid"], r["config"]) for r in data["rows"]}
        if len(done_keys) == len(items) * len(CONFIGS):
            continue
        rec = Recognizer(str(HF / "v4_mix" / "work" / f"fold{fold}.pt"),
                         decoder="beam", beam=24, lm_weight=0.0, nbest=1)
        t0 = time.time()
        for i, item in enumerate(items):
            if all((item["uid"], ci) in done_keys for ci in range(len(CONFIGS))):
                continue
            c = clip_map[(item["id"], item["phrase"])]
            with torch.no_grad():
                x = torch.from_numpy(features(c["span"], rec.feat))[None].to(rec.device)
                lens = torch.tensor([len(c["span"])], device=rec.device)
                logp = rec.model(x, lens)[0].cpu().log_softmax(-1).numpy()
            smoothed = {s: smooth_log_probs(logp, s) for s in (1, 3)}
            for ci, cfg in enumerate(CONFIGS):
                lp = smoothed[cfg["smooth"]]
                hyp = ctc_beam_search(lp, beam=cfg["beam"], topk=32, nbest=1,
                                      lm=None, lm_weight=0.0,
                                      word_bonus=cfg["word_bonus"])[0][0]
                data["rows"].append({"uid": item["uid"], "id": item["id"],
                                     "phrase": item["phrase"], "config": ci,
                                     "pred": [rec.names[j] for j in hyp]})
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            if (i + 1) % 10 == 0:
                print(f"fold {fold}: {i + 1}/{len(items)} [{time.time() - t0:.0f}s]",
                      flush=True)
        print(f"fold {fold} готов [{time.time() - t0:.0f}s]", flush=True)
    aggregate(selection)


if __name__ == "__main__":
    main()
