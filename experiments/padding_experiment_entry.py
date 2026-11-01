# -*- coding: utf-8 -*-
"""Парный sweep контекста вокруг cue: pad 0/0.4/0.8/1.2 с, shift=0.

Использует заранее отобранные held-out видео alignment_exp и fold-модели v4.
Непокрытые всеми вариантами клипы исключаются до распознавания из всех вариантов.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

import alignment_experiment_entry as base

HF = base.HF
ROOT = base.ROOT
OUT = HF / "padding_exp"
SELECTION = OUT / "selection.json"
REPORT = OUT / "report.json"
STATUS = OUT / "status.json"
LOCK = OUT / "experiment.lock"
PADS = (0.0, 0.4, 0.8, 1.2)


def prepare_selection():
    source = base.read_json(base.SELECTION, {})
    samples, rejected = [], 0
    for sample in source.get("samples", []):
        try:
            with np.load(base.CUES / f"{sample['id']}.npz") as d:
                n, t0 = len(d["frames"]), int(d["t0"])
            t1 = t0 + n
            if any(base.window_bounds(sample["cue_a"], sample["cue_b"],
                                      sample["fps"], 0.0, pad)[0] < t0 or
                   base.window_bounds(sample["cue_a"], sample["cue_b"],
                                      sample["fps"], 0.0, pad)[1] > t1
                   for pad in PADS):
                rejected += 1
                continue
            samples.append(sample)
        except (OSError, ValueError, KeyError):
            rejected += 1
    if len(samples) < 50:
        raise RuntimeError(f"Слишком мало клипов с общим покрытием: {len(samples)}")
    return {"method": "held-out VOGinfo, shift=0, общие ID для всех pad",
            "source_selection": str(base.SELECTION), "pads": PADS,
            "coverage_rejected": rejected, "samples": samples,
            "source_rows": {s["id"]: source["source_rows"][s["id"]] for s in samples}}


def prediction_row(sample, source_row, pad, pred, used):
    row = base.row_for_prediction(sample, source_row, 0.0, pred, used)
    row.pop("shift")
    row["pad"] = pad
    return row


def load_fold_rows(fold):
    rows = base.read_json(OUT / f"fold{fold}_rows.json", [])
    return {f'{r["pad"]:.2f}|{r["id"]}': r for r in rows}


def compare(rows_by_pad):
    groups = {pad: {r["id"] for r in rows} for pad, rows in rows_by_pad.items()}
    if any(ids != groups[0.8] for ids in groups.values()):
        raise RuntimeError("ID вариантов pad различаются: парное сравнение запрещено")
    metrics = {f"{pad:.2f}": base.metrics(rows) for pad, rows in rows_by_pad.items()}
    baseline = {r["id"]: r for r in rows_by_pad[0.8]}
    paired = {}
    for pad, rows in rows_by_pad.items():
        delta = [r["hits"] - baseline[r["id"]]["hits"] for r in rows]
        paired[f"{pad:.2f}"] = {"improved": sum(d > 0 for d in delta),
                               "worse": sum(d < 0 for d in delta),
                               "same": sum(d == 0 for d in delta)}
    return {"by_pad": metrics, "paired_vs_0.80": paired}


def status(phase, **extra):
    data = {"phase": phase, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    data.update(extra)
    base.write_json(STATUS, data)


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = base.read_json(SELECTION)
    if not selection or not selection.get("samples"):
        selection = prepare_selection()
        base.write_json(SELECTION, selection)
    samples = selection["samples"]
    source_rows = selection["source_rows"]
    expected = len(samples) * len(PADS)
    status("evaluating", selected=len(samples), expected_decodes=expected)
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 3500:
        return
    LOCK.write_text(str(time.time()), encoding="utf-8")
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from tools.dialogue_translator import Recognizer
        for fold in range(5):
            fold_samples = [s for s in samples if s["fold"] == fold]
            done = load_fold_rows(fold)
            todo = [(s, pad) for s in fold_samples for pad in PADS
                    if f"{pad:.2f}|{s['id']}" not in done]
            if todo:
                rec = Recognizer(str(HF / "v4_mix" / "work" / f"fold{fold}.pt"),
                                 decoder="beam", beam=24, lm_weight=0.0,
                                 word_bonus=3.0, smooth=1, nbest=1)
                n_new = 0
                for sample, pad in todo:
                    with np.load(base.CUES / f"{sample['id']}.npz") as d:
                        frames = d["frames"]
                    window = base.shifted_span(frames, sample["fps"], sample["cue_a"],
                                               sample["cue_b"], sample["t0"], 0.0, pad)
                    span, used = base.longest_hand_span(window, sample["fps"])
                    glosses, _ = rec.decode(span)
                    row = prediction_row(sample, source_rows[sample["id"]], pad,
                                         [g for g, _ in glosses], used)
                    done[f"{pad:.2f}|{sample['id']}"] = row
                    n_new += 1
                    if n_new % 10 == 0:
                        base.write_json(OUT / f"fold{fold}_rows.json", list(done.values()))
                        status("evaluating", selected=len(samples), expected_decodes=expected,
                               completed_decodes=sum(len(load_fold_rows(k)) for k in range(5)))
                base.write_json(OUT / f"fold{fold}_rows.json", list(done.values()))
            status("evaluating", selected=len(samples), expected_decodes=expected,
                   completed_decodes=sum(len(load_fold_rows(k)) for k in range(5)))
    finally:
        LOCK.unlink(missing_ok=True)

    rows_by_pad = {pad: [] for pad in PADS}
    for fold in range(5):
        for row in load_fold_rows(fold).values():
            rows_by_pad[float(row["pad"])].append(row)
    if any(len(rows) != len(samples) for rows in rows_by_pad.values()):
        status("evaluating", selected=len(samples), expected_decodes=expected,
               completed_decodes=sum(len(rows) for rows in rows_by_pad.values()))
        return
    comparison = compare(rows_by_pad)
    per_fold = {}
    for fold in range(5):
        per_fold[str(fold)] = {
            f"{pad:.2f}": base.metrics([r for r in rows if r["fold"] == fold])
            for pad, rows in rows_by_pad.items()}
    best = max(PADS, key=lambda p: (comparison["by_pad"][f"{p:.2f}"]["f1"],
                                    comparison["by_pad"][f"{p:.2f}"]["recall"],
                                    -abs(p - 0.8)))
    report = {"complete": True, "method": selection["method"], "clips": len(samples),
              "unique_videos": len({s["video"] for s in samples}),
              "coverage_rejected": selection["coverage_rejected"], "pads": PADS,
              "shift": 0.0, "comparison": comparison, "per_fold": per_fold,
              "best_exploratory_pad": best,
              "decision": ("текущий pad 0.8 не хуже; извлечение не менять" if best == 0.8
                           else "только кандидат: нужна полная проверка на новых видео"),
              "limitations": ["Разведочная выборка из alignment_exp",
                              "Фолды v4 разбиты по клипам, не по видео",
                              "Субтитры — слабые метки русского текста"],
              "production_changed": False}
    base.write_json(REPORT, report)
    status("done", selected=len(samples), report=str(REPORT), best_pad=best)


def run(ctx):
    if not ctx.get("_worker"):
        import subprocess
        OUT.mkdir(parents=True, exist_ok=True)
        log_path = OUT / "experiment.log"
        try:
            with open(log_path, "ab") as logf:
                proc = subprocess.run(
                    [r"D:\venvs\gesturelab\Scripts\python.exe", str(Path(__file__)), "--worker"],
                    cwd=str(HF), env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                    stdout=logf, stderr=subprocess.STDOUT, timeout=3300)
        except subprocess.TimeoutExpired:
            proc = None
        if proc is not None and proc.returncode != 0:
            status("error", returncode=proc.returncode, log=str(log_path))
            raise RuntimeError(f"Worker завершился с кодом {proc.returncode}; см. {log_path}")
        report = base.read_json(REPORT, {})
        if report.get("complete"):
            return {"artifact": {"summary": "Sweep ширины окна завершён", "complete": True,
                                 "report": str(REPORT), "comparison": report["comparison"]}}
        return {"artifact": {"summary": "Срез sweep завершён; продолжение по расписанию",
                             "complete": False, "status": base.read_json(STATUS, {})}}
    evaluate()
    report = base.read_json(REPORT, {})
    return {"artifact": {"summary": "Worker sweep ширины окна завершён",
                         "complete": bool(report.get("complete"))}}


def selftest():
    rows = {p: [{"id": "v_000", "hits": int(p == 0.8), "total": 1,
                 "pred": ["слово"] if p == 0.8 else []}] for p in PADS}
    result = compare(rows)
    assert result["by_pad"]["0.80"]["recall"] == 1.0
    assert result["paired_vs_0.80"]["0.40"]["worse"] == 1


if __name__ == "__main__":
    import sys
    if "--worker" in sys.argv:
        run({"_worker": True})
    else:
        selftest()
        print("padding experiment selftest ok")
