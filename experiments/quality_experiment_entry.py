# -*- coding: utf-8 -*-
"""Парный CV-эксперимент: исходные метки против фильтра качества только в train."""
import json
import os
import subprocess
import time
from collections import defaultdict

PY = r"D:\venvs\gesturelab\Scripts\python.exe"
HF = r"D:\hfdata"
EXP = os.path.join(HF, "quality_exp")
STATUS = os.path.join(EXP, "status.json")
LOCK = os.path.join(EXP, "experiment.lock")
TRAIN_BUDGET_S = 3300
FOLDS = 2
VARIANTS = ("baseline", "filtered")


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return default


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def report_path(variant):
    return os.path.join(EXP, variant, "report.json")


def complete_report(variant):
    rep = read_json(report_path(variant), {})
    return bool(rep and not rep.get("partial") and rep.get("held_beam") and rep.get("rows"))


def folds_done(variant):
    progress = read_json(os.path.join(EXP, variant, "work", "progress.json"), {})
    return len(progress.get("folds", {}))


def source_of(cid, rutube_ids):
    if cid in rutube_ids:
        return "rutube"
    if cid.startswith(("train_", "dev_", "test_")):
        return "SP-10"
    return "VOGinfo"


def metrics(rows):
    hits = sum(int(r.get("hits", 0)) for r in rows)
    tokens = sum(int(r.get("total", 0)) for r in rows)
    predicted = sum(len(r.get("pred", [])) for r in rows)
    recall = hits / max(tokens, 1)
    precision = hits / max(predicted, 1)
    return {
        "clips": len(rows),
        "hits": hits,
        "tokens": tokens,
        "predicted_tokens": predicted,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 6),
        "empty": round(sum(not r.get("pred") for r in rows) / max(len(rows), 1), 6),
    }


def slovo_metrics(rep):
    rows = [f["slovo_clean"] for f in rep.get("fold_report", []) if f.get("slovo_clean")]
    keys = ("gloss_acc", "exact_acc", "edit_rate")
    return {k: round(sum(r[k] for r in rows) / len(rows), 6)
            for k in keys} if rows else {}


def compare_reports(baseline, filtered, rutube_ids):
    b_by_id = {r["id"]: r for r in baseline["rows"]}
    f_by_id = {r["id"]: r for r in filtered["rows"]}
    if set(b_by_id) != set(f_by_id):
        raise RuntimeError("Held-out ID baseline и filtered различаются: парное сравнение запрещено")
    ids = sorted(b_by_id)
    groups = defaultdict(list)
    for cid in ids:
        groups[source_of(cid, rutube_ids)].append(cid)
    groups["all"] = ids
    out = {}
    for name, group in groups.items():
        b = metrics([b_by_id[cid] for cid in group])
        f = metrics([f_by_id[cid] for cid in group])
        out[name] = {"baseline": b, "filtered": f,
                     "delta": {k: round(f[k] - b[k], 6)
                               for k in ("hits", "recall", "precision", "f1", "empty")}}
    hit_delta = [f_by_id[cid].get("hits", 0) - b_by_id[cid].get("hits", 0) for cid in ids]
    paired = {"clips": len(ids), "improved": sum(d > 0 for d in hit_delta),
              "worse": sum(d < 0 for d in hit_delta), "same": sum(d == 0 for d in hit_delta)}
    b_slovo = slovo_metrics(baseline)
    f_slovo = slovo_metrics(filtered)
    return {
        "method": "2-fold paired held-out; train_ids применён только к filtered train",
        "identical_held_ids": True,
        "paired_hit_changes": paired,
        "by_source": out,
        "slovo_clean": {"baseline": b_slovo, "filtered": f_slovo,
                        "delta": {k: round(f_slovo[k] - b_slovo[k], 6) for k in b_slovo}},
        "note": "Эксплораторный эксперимент на 2000 клипов; production автоматически не меняется.",
    }


def command(variant):
    work = os.path.join(EXP, variant, "work")
    os.makedirs(work, exist_ok=True)
    cmd = [PY, os.path.join(HF, "GestureLab_PC_v0.1", "tools", "finetune_ctc_phrases.py"),
           "--inventory", "D:/hfdata/v4_mix/v4_mix_inventory.json",
           "--cache", "D:/hfdata/v4_mix/landmarks",
           "--init", "D:/hfdata/rsl_translator_v1/prod_v4.pt",
           "--work", work, "--out", report_path(variant),
           "--folds", str(FOLDS), "--epochs", "6", "--steps", "40", "--limit", "2000",
           "--no-canon", "--aug", "hard", "--recombine-frac", "0.25",
           "--dup-times", "8", "--dup-ids", "D:/hfdata/combined_phrases/rutube_ids.json",
           "--align-from", "D:/hfdata/phrase_combined_dup/fold{fold}.pt",
           "--resume", "--no-llm"]
    if variant == "filtered":
        cmd += ["--train-ids", "D:/hfdata/quality_train_ids.json"]
    return cmd


def status(phase, **extra):
    data = {"phase": phase,
            "folds": {v: folds_done(v) for v in VARIANTS},
            "complete": {v: complete_report(v) for v in VARIANTS},
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    data.update(extra)
    write_json(STATUS, data)
    return data


def selftest():
    rows_a = [{"id": "train_1", "hits": 1, "total": 2, "pred": ["я"]},
              {"id": "video_000", "hits": 0, "total": 1, "pred": []}]
    rows_b = [{"id": "train_1", "hits": 2, "total": 2, "pred": ["я", "дом"]},
              {"id": "video_000", "hits": 1, "total": 1, "pred": ["день"]}]
    slovo = {"gloss_acc": 0.5, "exact_acc": 0.4, "edit_rate": 0.2}
    fold = {"slovo_clean": slovo}
    a = {"rows": rows_a, "fold_report": [fold]}
    b = {"rows": rows_b, "fold_report": [fold]}
    result = compare_reports(a, b, set())
    assert result["identical_held_ids"]
    assert result["by_source"]["all"]["delta"]["hits"] == 2
    assert result["by_source"]["SP-10"]["filtered"]["recall"] == 1.0
    assert result["paired_hit_changes"] == {"clips": 2, "improved": 2, "worse": 0, "same": 0}
    bad = {"rows": rows_b[:1], "fold_report": [fold]}
    try:
        compare_reports(a, bad, set())
    except RuntimeError:
        pass
    else:
        raise AssertionError("Разные held-out ID должны запрещать сравнение")


def run(ctx):
    os.makedirs(EXP, exist_ok=True)
    if all(complete_report(v) for v in VARIANTS):
        rutube_ids = set(read_json(os.path.join(HF, "combined_phrases", "rutube_ids.json"), []))
        comparison = compare_reports(read_json(report_path("baseline")),
                                     read_json(report_path("filtered")), rutube_ids)
        write_json(os.path.join(EXP, "comparison.json"), comparison)
        status("done", comparison=os.path.join(EXP, "comparison.json"))
        return {"artifact": {"summary": "Парный эксперимент завершён; production не изменена",
                             "phase": "done", "complete": True,
                             "comparison": comparison}}

    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 3500:
        status("busy")
        return {"artifact": {"summary": "Предыдущий срез эксперимента ещё идёт",
                             "phase": "busy", "complete": False,
                             "folds": {v: folds_done(v) for v in VARIANTS}}}

    for variant in VARIANTS:
        if complete_report(variant):
            continue
        with open(LOCK, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
        log_path = os.path.join(EXP, variant, "train.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            with open(log_path, "ab") as logf:
                try:
                    proc = subprocess.run(command(variant), cwd=os.path.join(HF, "GestureLab_PC_v0.1"),
                                          env=env, stdout=logf, stderr=subprocess.STDOUT,
                                          timeout=TRAIN_BUDGET_S)
                except subprocess.TimeoutExpired:
                    proc = None
        finally:
            try:
                os.unlink(LOCK)
            except FileNotFoundError:
                pass
        if proc is not None and proc.returncode != 0:
            status("error", variant=variant, returncode=proc.returncode, log=log_path)
            raise RuntimeError(f"{variant}: код {proc.returncode}; см. {log_path}")
        current = status("training", variant=variant, log=log_path)
        return {"artifact": {"summary": f"{variant}: фолды {folds_done(variant)}/{FOLDS}",
                             "phase": "training", "variant": variant, "complete": False,
                             "folds": current["folds"],
                             "reports_complete": current["complete"]}}

    raise RuntimeError("Неожиданное состояние: отчёты не готовы, но варианты не выбраны")


if __name__ == "__main__":
    selftest()
    print("quality experiment selftest ok")
