# -*- coding: utf-8 -*-
"""Парная проверка сдвига русских субтитров относительно жестов ВОГinfo.

Только готовые fold-модели v4 и их собственные held-out строки. Все сдвиги
оцениваются на одних ID; production и обучение v5 этот скрипт не изменяет.
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HF = Path(r"D:\hfdata")
ROOT = HF / "GestureLab_PC_v0.1"
CUES = HF / "ytsl25_rsl" / "cues"
SUBS = HF / "ytsl25_rsl" / "subs"
PROGRESS = HF / "v4_mix" / "work" / "progress.json"
OUT = HF / "alignment_exp"
SELECTION = OUT / "selection.json"
REPORT = OUT / "report.json"
STATUS = OUT / "status.json"
LOCK = OUT / "experiment.lock"
SHIFTS = (-0.5, -0.25, 0.0, 0.25, 0.5)
PAD = 0.8
PER_FOLD = 23
SEED = 20261031


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def parse_vtt(path):
    """Вернуть [(start, end, text)]; реализация совместима с ytsl25_extract."""
    import re
    spans, cur = [], None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(
                r"(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> "
                r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", line)
            if m:
                g = [int(x) for x in m.groups()]
                cur = [g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                       g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000, []]
                spans.append(cur)
            elif cur is not None and line.strip() and "-->" not in line:
                cur[2].append(line.strip())
    out = []
    for a, b, tx in spans:
        text = re.sub(r"<[^>]+>", "", " ".join(tx)).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            out.append((a, b, text))
    return out


def window_bounds(cue_a, cue_b, fps, shift, pad=PAD):
    """Абсолютные кадры окна cue±pad после сдвига, end не включается."""
    a = int(round((cue_a + shift - pad) * fps))
    b = int(round((cue_b + shift + pad) * fps))
    if b <= a:
        raise ValueError("Пустое временное окно")
    return a, b


def shifted_span(frames, fps, cue_a, cue_b, t0, shift, pad=PAD):
    """Вырезать окно только при полном покрытии; незаметного укорачивания нет."""
    a, b = window_bounds(cue_a, cue_b, fps, shift, pad)
    t0 = int(t0)
    t1 = t0 + len(frames)
    if a < t0 or b > t1:
        raise ValueError(f"Окно [{a}, {b}) вне кэша [{t0}, {t1})")
    return frames[a - t0:b - t0]


def longest_hand_span(frames, fps):
    """Тот же выбор реплики, что в load_phrase_clips."""
    import sys
    sys.path.insert(0, str(ROOT))
    from sign_utils import split_utterances
    spans = split_utterances(frames, fps, pause=0.7, min_len=0.4, max_len=15.0)
    a, b = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(frames))
    return frames[a:b].astype(np.float32), [int(a), int(b)]


def is_voginfo(cid):
    return not cid.startswith(("train_", "dev_", "test_")) and "_" in cid \
        and cid.rsplit("_", 1)[1].isdigit()


def sample_record(cid, fold):
    """Метаданные одного cue; None, если любой сдвиг не покрыт кэшем."""
    path = CUES / f"{cid}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as d:
            fps = float(d["fps"])
            t0, t1 = int(d["t0"]), int(d["t1"])
            n = len(d["frames"])
            video = str(d["video"])
        if t1 - t0 != n:
            return None
        ci = int(cid.rsplit("_", 1)[1])
        cues = parse_vtt(SUBS / f"{video}.ru.vtt")
        if ci >= len(cues):
            return None
        cue_a, cue_b, _ = cues[ci]
        bounds = {}
        for shift in SHIFTS:
            a, b = window_bounds(cue_a, cue_b, fps, shift)
            if a < t0 or b > t1:
                return None
            bounds[f"{shift:+.2f}"] = [a, b]
        return {"id": cid, "fold": fold, "video": video, "cue_index": ci,
                "cue_a": cue_a, "cue_b": cue_b, "fps": fps,
                "t0": t0, "t1": t1, "bounds": bounds}
    except (OSError, ValueError, KeyError, IndexError):
        return None


def choose_samples(progress, per_fold=PER_FOLD):
    """Детерминированно: до N на фолд, одно видео во всей выборке один раз."""
    rng = np.random.default_rng(SEED)
    selected, rejected, used_videos = [], 0, set()
    for fold in range(5):
        rows = progress.get("folds", {}).get(str(fold), {}).get("rows", [])
        ids = sorted({r["id"] for r in rows if is_voginfo(r["id"])})
        ids = [ids[i] for i in rng.permutation(len(ids))]
        got = 0
        for cid in ids:
            rec = sample_record(cid, fold)
            if rec is None:
                rejected += 1
                continue
            if rec["video"] in used_videos:
                continue
            selected.append(rec)
            used_videos.add(rec["video"])
            got += 1
            if got >= per_fold:
                break
    return selected, {"all_shift_coverage_rejected": rejected,
                      "unique_videos": len(used_videos)}


def row_for_prediction(sample, source_row, shift, pred, used):
    from tools.finetune_ctc_phrases import matched
    return {"id": sample["id"], "fold": sample["fold"], "video": sample["video"],
            "shift": shift, "phrase": source_row["phrase"], "canon": source_row["canon"],
            "pred": pred, "hits": sum(matched(pred, source_row["canon"]).values()),
            "total": int(source_row["total"]), "predicted_tokens": len(pred),
            "used": used}


def metrics(rows):
    hits = sum(int(r["hits"]) for r in rows)
    tokens = sum(int(r["total"]) for r in rows)
    predicted = sum(len(r.get("pred", [])) for r in rows)
    recall = hits / max(tokens, 1)
    precision = hits / max(predicted, 1)
    return {"clips": len(rows), "hits": hits, "tokens": tokens,
            "predicted_tokens": predicted, "recall": round(recall, 6),
            "precision": round(precision, 6),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 6),
            "empty": round(sum(not r.get("pred") for r in rows) / max(len(rows), 1), 6)}


def compare(rows_by_shift):
    ids = {shift: {r["id"] for r in rows} for shift, rows in rows_by_shift.items()}
    if any(group != ids[0.0] for group in ids.values()):
        raise RuntimeError("ID сдвигов различаются: парное сравнение запрещено")
    by_shift = {f"{s:+.2f}": metrics(rows_by_shift[s]) for s in SHIFTS}
    base = {r["id"]: r for r in rows_by_shift[0.0]}
    paired = {}
    for shift in SHIFTS:
        delta = [r["hits"] - base[r["id"]]["hits"] for r in rows_by_shift[shift]]
        paired[f"{shift:+.2f}"] = {"improved": sum(d > 0 for d in delta),
                                  "worse": sum(d < 0 for d in delta),
                                  "same": sum(d == 0 for d in delta)}
    return {"by_shift": by_shift, "paired_vs_zero": paired}


def load_fold_rows(fold):
    rows = read_json(OUT / f"fold{fold}_rows.json", [])
    return {f'{r["shift"]:+.2f}|{r["id"]}': r for r in rows}


def status(phase, **extra):
    data = {"phase": phase, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    data.update(extra)
    write_json(STATUS, data)


def run(ctx):
    import sys
    if not ctx.get("_worker"):
        # В automation-рантайме нет cv2/torch проекта: вычисления только в его venv.
        import subprocess
        OUT.mkdir(parents=True, exist_ok=True)
        log_path = OUT / "experiment.log"
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            with open(log_path, "ab") as logf:
                proc = subprocess.run(
                    [r"D:\venvs\gesturelab\Scripts\python.exe", str(Path(__file__)), "--worker"],
                    cwd=str(HF), env=env, stdout=logf, stderr=subprocess.STDOUT,
                    timeout=3300)
        except subprocess.TimeoutExpired:
            proc = None
        if proc is not None and proc.returncode != 0:
            status("error", returncode=proc.returncode, log=str(log_path))
            raise RuntimeError(f"Worker завершился с кодом {proc.returncode}; см. {log_path}")
        report = read_json(REPORT, {})
        if report.get("complete"):
            return {"artifact": {"summary": "Парный sweep сдвигов завершён", "complete": True,
                                 "report": str(REPORT), "comparison": report["comparison"]}}
        state = read_json(STATUS, {})
        return {"artifact": {"summary": "Срез sweep завершён; продолжение по расписанию",
                             "complete": False, "status": state}}
    sys.path.insert(0, str(ROOT))
    OUT.mkdir(parents=True, exist_ok=True)
    old_report = read_json(REPORT)
    if old_report and old_report.get("complete"):
        return {"artifact": {"summary": "Парный sweep сдвигов завершён", "complete": True,
                             "report": str(REPORT), "comparison": old_report["comparison"]}}

    selection = read_json(SELECTION)
    if not selection or not selection.get("samples"):
        progress = read_json(PROGRESS, {})
        samples, coverage = choose_samples(progress)
        if len(samples) < PER_FOLD * 5:
            raise RuntimeError(f"Недостаточно общих клипов: {len(samples)} из {PER_FOLD * 5}")
        source_rows = {r["id"]: r for f in progress["folds"].values() for r in f["rows"]}
        selection = {"method": f"{len(samples)} held-out VOGinfo, {PER_FOLD}/fold, "
                               "одно видео один раз",
                     "seed": SEED, "shifts": SHIFTS, "pad": PAD,
                     "coverage": coverage, "samples": samples,
                     "source_rows": {s["id"]: source_rows[s["id"]] for s in samples}}
        write_json(SELECTION, selection)
    samples = selection["samples"]
    source_rows = selection["source_rows"]
    expected = len(samples) * len(SHIFTS)
    status("evaluating", selected=len(samples), expected_decodes=expected)

    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 3500:
        return {"artifact": {"summary": "Предыдущий срез sweep ещё идёт", "complete": False}}
    LOCK.write_text(str(time.time()), encoding="utf-8")
    try:
        from tools.dialogue_translator import Recognizer
        for fold in range(5):
            fold_samples = [s for s in samples if s["fold"] == fold]
            done = load_fold_rows(fold)
            todo = [(s, shift) for s in fold_samples for shift in SHIFTS
                    if f"{shift:+.2f}|{s['id']}" not in done]
            if todo:
                rec = Recognizer(str(HF / "v4_mix" / "work" / f"fold{fold}.pt"),
                                 decoder="beam", beam=24, lm_weight=0.0,
                                 word_bonus=3.0, smooth=1, nbest=1)
                n_new = 0
                for sample, shift in todo:
                    with np.load(CUES / f"{sample['id']}.npz") as d:
                        frames = d["frames"]
                    window = shifted_span(frames, sample["fps"], sample["cue_a"],
                                          sample["cue_b"], sample["t0"], shift)
                    span, used = longest_hand_span(window, sample["fps"])
                    glosses, _ = rec.decode(span)
                    row = row_for_prediction(sample, source_rows[sample["id"]], shift,
                                             [g for g, _ in glosses], used)
                    done[f"{shift:+.2f}|{sample['id']}"] = row
                    n_new += 1
                    if n_new % 10 == 0:
                        write_json(OUT / f"fold{fold}_rows.json", list(done.values()))
                        status("evaluating", selected=len(samples), expected_decodes=expected,
                               completed_decodes=sum(len(load_fold_rows(k)) for k in range(5)))
                write_json(OUT / f"fold{fold}_rows.json", list(done.values()))
            status("evaluating", selected=len(samples), expected_decodes=expected,
                   completed_decodes=sum(len(load_fold_rows(k)) for k in range(5)),
                   folds_complete=sum(len(load_fold_rows(k)) == PER_FOLD * len(SHIFTS)
                                      for k in range(5)))
    finally:
        LOCK.unlink(missing_ok=True)

    rows_by_shift = {shift: [] for shift in SHIFTS}
    for fold in range(5):
        for row in load_fold_rows(fold).values():
            rows_by_shift[float(row["shift"])].append(row)
    if any(len(rows) != len(samples) for rows in rows_by_shift.values()):
        status("evaluating", selected=len(samples), expected_decodes=expected,
               completed_decodes=sum(len(v) for v in rows_by_shift.values()))
        return {"artifact": {"summary": "Sweep продолжится в следующем срезе", "complete": False}}
    comparison = compare(rows_by_shift)
    per_fold = {}
    for fold in range(5):
        subset = {s: [r for r in rows if r["fold"] == fold] for s, rows in rows_by_shift.items()}
        per_fold[str(fold)] = {f"{s:+.2f}": metrics(rows) for s, rows in subset.items()}
    best = max(SHIFTS, key=lambda s: (comparison["by_shift"][f"{s:+.2f}"]["f1"],
                                      comparison["by_shift"][f"{s:+.2f}"]["recall"],
                                      -abs(s)))
    report = {"complete": True, "method": selection["method"], "clips": len(samples),
              "unique_videos": selection["coverage"]["unique_videos"], "shifts": SHIFTS,
              "pad_seconds": PAD, "decoder": {"beam": 24, "lm_weight": 0.0,
                                               "word_bonus": 3.0, "smooth": 1},
              "comparison": comparison, "per_fold": per_fold,
              "best_exploratory_shift": best,
              "decision": ("сдвиг 0 не хуже остальных; менять извлечение не нужно" if best == 0
                           else "только кандидат: нужен полный group-CV по видео, production не менять"),
              "limitations": ["Разведочная выборка, а не полный CV",
                              "Фолды v4 разбиты по клипам, не по видео",
                              "Русские субтитры — слабые метки, не точная глосс-разметка"],
              "production_changed": False}
    write_json(REPORT, report)
    status("done", selected=len(samples), report=str(REPORT), best_shift=best)
    return {"artifact": {"summary": f"Sweep завершён; лучший разведочный сдвиг {best:+.2f} с",
                         "complete": True, "report": str(REPORT),
                         "best_shift": best, "comparison": comparison}}


def selftest():
    frames = np.arange(1000, dtype=np.float32).reshape(100, 10)
    got = shifted_span(frames, 25, 2.0, 2.4, 0, 0.0, pad=0.8)
    assert len(got) == 50 and got[0, 0] == 300 and got[-1, 0] == 790, \
        (len(got), got[0, 0], got[-1, 0])
    try:
        shifted_span(frames, 25, 0.0, 1.0, 0, -0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Непокрытое окно должно отклоняться")
    assert window_bounds(2.0, 2.4, 25, 0.5, 0.8) == (42, 92)
    rows = {s: [{"id": "v_000", "hits": int(s == 0), "total": 1,
                 "pred": ["слово"] if s == 0 else []}] for s in SHIFTS}
    result = compare(rows)
    assert result["by_shift"]["+0.00"]["recall"] == 1.0
    assert result["paired_vs_zero"]["+0.25"]["worse"] == 1
    assert is_voginfo("abc_000") and not is_voginfo("train_410")


if __name__ == "__main__":
    import sys
    if "--worker" in sys.argv:
        run({"_worker": True})
    else:
        selftest()
        print("alignment experiment selftest ok")
