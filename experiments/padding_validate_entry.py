# -*- coding: utf-8 -*-
"""Полная парная валидация pad=0 против pad=0.8 на остальных held-out клипах v4.

115 клипов разведочного padding_exp исключены. Разбивка и модели остаются v4;
production, текущее извлечение и v5 не изменяются.
"""
import time
from pathlib import Path

import numpy as np

import alignment_experiment_entry as base
import padding_experiment_entry as padding

HF = base.HF
OUT = HF / "padding_validate"
SELECTION = OUT / "selection.json"
REPORT = OUT / "report.json"
STATUS = OUT / "status.json"
LOCK = OUT / "experiment.lock"
PADS = (0.0, 0.8)


def metadata_record(cid, fold, cues_by_video):
    path = base.CUES / f"{cid}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as d:
            fps, t0, t1 = float(d["fps"]), int(d["t0"]), int(d["t1"])
            video = str(d["video"])
        ci = int(cid.rsplit("_", 1)[1])
        if video not in cues_by_video:
            cues_by_video[video] = base.parse_vtt(base.SUBS / f"{video}.ru.vtt")
        cues = cues_by_video[video]
        if ci >= len(cues):
            return None
        cue_a, cue_b, _ = cues[ci]
        for pad in PADS:
            a, b = base.window_bounds(cue_a, cue_b, fps, 0.0, pad)
            if a < t0 or b > t1:
                return None
        return {"id": cid, "fold": fold, "video": video, "cue_index": ci,
                "cue_a": cue_a, "cue_b": cue_b, "fps": fps, "t0": t0, "t1": t1}
    except (OSError, ValueError, KeyError, IndexError):
        return None


def prepare_selection():
    progress = base.read_json(base.PROGRESS, {})
    exploratory = base.read_json(base.SELECTION, {})
    excluded = {s["id"] for s in exploratory.get("samples", [])}
    samples, source_rows, rejected = [], {}, 0
    cues_by_video = {}
    for fold in range(5):
        rows = progress.get("folds", {}).get(str(fold), {}).get("rows", [])
        for source in rows:
            cid = source["id"]
            if cid in excluded or not base.is_voginfo(cid):
                continue
            rec = metadata_record(cid, fold, cues_by_video)
            if rec is None:
                rejected += 1
                continue
            samples.append(rec)
            source_rows[cid] = {k: source[k] for k in ("phrase", "canon", "total")}
    if len(samples) < 1000:
        raise RuntimeError(f"Недостаточно клипов полной валидации: {len(samples)}")
    return {"method": "все held-out VOGinfo v4, кроме 115 exploratory; общие ID pad 0/0.8",
            "pads": PADS, "coverage_rejected": rejected,
            "exploratory_excluded": len(excluded), "samples": samples,
            "source_rows": source_rows}


def load_fold_rows(fold):
    rows = base.read_json(OUT / f"fold{fold}_rows.json", [])
    return {f'{r["pad"]:.2f}|{r["id"]}': r for r in rows}


def status(phase, **extra):
    data = {"phase": phase, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    data.update(extra)
    base.write_json(STATUS, data)


def counts():
    return [len(load_fold_rows(k)) for k in range(5)]


def evaluate():
    import sys
    sys.path.insert(0, str(base.ROOT))
    OUT.mkdir(parents=True, exist_ok=True)
    selection = base.read_json(SELECTION)
    if not selection or not selection.get("samples"):
        selection = prepare_selection()
        base.write_json(SELECTION, selection)
    samples = selection["samples"]
    source_rows = selection["source_rows"]
    expected = len(samples) * len(PADS)
    status("evaluating", selected=len(samples), expected_decodes=expected,
           completed_decodes=sum(counts()))
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 3500:
        return
    LOCK.write_text(str(time.time()), encoding="utf-8")
    try:
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
                    row = padding.prediction_row(sample, source_rows[sample["id"]], pad,
                                                 [g for g, _ in glosses], used)
                    done[f"{pad:.2f}|{sample['id']}"] = row
                    n_new += 1
                    if n_new % 50 == 0:
                        base.write_json(OUT / f"fold{fold}_rows.json", list(done.values()))
                        status("evaluating", selected=len(samples), expected_decodes=expected,
                               completed_decodes=sum(counts()), current_fold=fold)
                base.write_json(OUT / f"fold{fold}_rows.json", list(done.values()))
            status("evaluating", selected=len(samples), expected_decodes=expected,
                   completed_decodes=sum(counts()), current_fold=fold)
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
    comparison = padding.compare(rows_by_pad)
    per_fold = {str(fold): {f"{pad:.2f}": base.metrics(
        [r for r in rows if r["fold"] == fold]) for pad, rows in rows_by_pad.items()}
        for fold in range(5)}
    candidate = comparison["by_pad"]["0.00"]
    baseline = comparison["by_pad"]["0.80"]
    improved = candidate["f1"] > baseline["f1"] and candidate["hits"] >= baseline["hits"]
    report = {"complete": True, "method": selection["method"], "clips": len(samples),
              "unique_videos": len({s["video"] for s in samples}),
              "coverage_rejected": selection["coverage_rejected"],
              "exploratory_clips_excluded": len(base.read_json(base.SELECTION)["samples"]),
              "pads": PADS, "shift": 0.0, "comparison": comparison, "per_fold": per_fold,
              "candidate_confirmed": improved,
              "decision": ("pad 0 подтверждён как кандидат для отдельного paired retrain; "
                           "production пока не менять" if improved else
                           "преимущество pad 0 не подтверждено; текущий рецепт не менять"),
              "limitations": ["Клипные фолды v4, не group-CV по видео",
                              "Русские субтитры — слабые метки, не глоссы",
                              "Это проверка окон на готовых моделях, не результат переобучения"],
              "production_changed": False}
    base.write_json(REPORT, report)
    status("done", selected=len(samples), report=str(REPORT), candidate_confirmed=improved)


def run(ctx):
    if not ctx.get("_worker"):
        import os
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
            # subprocess.run убил worker: его lock уже не означает живой процесс.
            LOCK.unlink(missing_ok=True)
            proc = None
        if proc is not None and proc.returncode != 0:
            status("error", returncode=proc.returncode, log=str(log_path))
            raise RuntimeError(f"Worker завершился с кодом {proc.returncode}; см. {log_path}")
        report = base.read_json(REPORT, {})
        return {"artifact": {"summary": "Полная валидация pad завершена" if report.get("complete")
                             else "Срез валидации завершён; продолжение по расписанию",
                             "complete": bool(report.get("complete")),
                             "comparison": report.get("comparison"),
                             "status": base.read_json(STATUS, {})}}
    evaluate()
    report = base.read_json(REPORT, {})
    return {"artifact": {"summary": "Worker полной валидации pad завершён",
                         "complete": bool(report.get("complete"))}}


def selftest():
    rows = {p: [{"id": "v_000", "hits": int(p == 0.0), "total": 1,
                 "pred": ["слово"] if p == 0.0 else []}] for p in PADS}
    result = padding.compare(rows)
    assert result["by_pad"]["0.00"]["recall"] == 1.0
    assert result["paired_vs_0.80"]["0.00"]["improved"] == 1


if __name__ == "__main__":
    import sys
    if "--worker" in sys.argv:
        run({"_worker": True})
    else:
        selftest()
        print("padding validation selftest ok")
