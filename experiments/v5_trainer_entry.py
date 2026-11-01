# -*- coding: utf-8 -*-
"""Автообучение CTC v5 на полном снимке ВОГinfo, с resume по фолдам/эпохам."""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import time

PY = r"D:\venvs\gesturelab\Scripts\python.exe"
HF = r"D:\hfdata"
CUES = os.path.join(HF, "ytsl25_rsl", "cues")
MIX = os.path.join(HF, "v5_mix")
WORK = os.path.join(MIX, "work")
WORK_PROD = os.path.join(MIX, "work_prod")
STATUS = os.path.join(HF, "v5_status.json")
NEEDED = 40000
FOLDS_TOTAL = 5
TRAIN_BUDGET_S = 3300
LOCK = os.path.join(MIX, "trainer.lock")


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return default


def write_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    os.replace(tmp, path)


def count_cues():
    n, done_v = 0, 0
    for f in glob.glob(os.path.join(CUES, "*.blocks.json")):
        st = read_json(f, {})
        if st.get("done"):
            done_v += 1
            n += len(st.get("meta", []))
    return n, done_v


def eligible_videos():
    probe = read_json(os.path.join(HF, "ytsl25_rsl_probe.json"), {})
    root = os.path.join(HF, "ytsl25_rsl")
    eligible = []
    for vid, rec in probe.items():
        mp4 = os.path.join(root, "videos", vid + ".mp4")
        vtt = os.path.join(root, "subs", vid + ".ru.vtt")
        if (rec.get("status") == "ok" and os.path.isfile(vtt)
                and os.path.isfile(mp4) and os.path.getsize(mp4) > 100000):
            eligible.append(vid)
    return eligible


def write_status(**kw):
    base = read_json(STATUS, {})
    base.update(kw)
    base["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(STATUS, base)


def progress_folds():
    rep = read_json(os.path.join(WORK, "progress.json"), {})
    return len(rep.get("folds", {}))


def done_report_exists():
    rep = read_json(os.path.join(MIX, "v5_cv.json"))
    return bool(rep and not rep.get("partial") and "held_beam" in rep)


def prod_report():
    rep = read_json(os.path.join(MIX, "v5_prod.json"))
    return rep if rep and rep.get("prod") else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_prod(rep):
    src = rep["prod"]
    expected = rep.get("prod_sha256")
    actual = sha256(src)
    if expected and actual != expected:
        raise RuntimeError("SHA-256 исходного prod_v5 не совпадает с отчётом")
    dst = os.path.join(HF, "rsl_translator_v1", "prod_v5.pt")
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    if sha256(tmp) != actual:
        raise RuntimeError("SHA-256 опубликованного prod_v5 не совпадает")
    os.replace(tmp, dst)
    return actual


def artifact(phase, cues, done_v, eligible, folds_done, complete, summary, **extra):
    data = {"summary": summary, "phase": phase, "cues": cues,
            "done_videos": done_v, "eligible_videos": eligible,
            "folds_done": folds_done, "folds_total": FOLDS_TOTAL,
            "complete": complete}
    data.update(extra)
    return {"artifact": data}


def make_snapshot(env):
    marker = os.path.join(MIX, "snapshot.ok")
    if os.path.exists(marker):
        return
    os.makedirs(os.path.join(MIX, "landmarks"), exist_ok=True)
    subprocess.run([PY, os.path.join(HF, "ytsl25_inventory.py")],
                   cwd=HF, env=env, check=True, capture_output=True, timeout=600)
    a = read_json(os.path.join(HF, "combined_phrases", "combined_inventory.json"))
    b = read_json(os.path.join(HF, "ytsl25_rsl", "ytsl25_inventory.json"))
    if not a or not b:
        raise RuntimeError("Не удалось прочитать инвентари для снимка v5")
    linked = 0
    for src_dir, inv in [(os.path.join(HF, "combined_phrases", "landmarks"), a),
                         (CUES, b)]:
        for phrase in inv["phrases"]:
            src = os.path.join(src_dir, phrase["id"] + ".npz")
            dst = os.path.join(MIX, "landmarks", phrase["id"] + ".npz")
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    os.link(src, dst)
                    linked += 1
                except OSError:
                    shutil.copyfile(src, dst)
                    linked += 1
    merged = {"channel": "v5: combined_phrases + full ytsl25_rsl VOGinfo cues",
              "phrases": a["phrases"] + b["phrases"]}
    write_json(os.path.join(MIX, "v5_mix_inventory.json"), merged)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(str(linked))


def run(ctx):
    cues, done_v = count_cues()
    eligible = len(eligible_videos())
    folds_done = progress_folds()
    rep_prod = prod_report()
    common = (cues, done_v, eligible)

    if done_report_exists() and rep_prod:
        digest = publish_prod(rep_prod)
        write_status(phase="done", cues=cues, done_videos=done_v,
                     eligible_videos=eligible, folds_done=FOLDS_TOTAL,
                     prod_sha256=digest)
        return artifact("done", *common, FOLDS_TOTAL, True,
                        "v5 ГОТОВА: CV 5/5, prod_v5.pt опубликован",
                        prod_sha256=digest)

    is_prod = done_report_exists()
    corpus_ready = cues >= NEEDED or (eligible > 0 and done_v >= eligible)
    if not corpus_ready:
        write_status(phase="waiting", cues=cues, done_videos=done_v,
                     eligible_videos=eligible, folds_done=folds_done)
        return artifact("waiting", *common, folds_done, False,
                        f"v5: корпус {cues}/{NEEDED}; видео {done_v}/{eligible}")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(WORK_PROD, exist_ok=True)
    make_snapshot(env)

    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 3500:
        return artifact("prod" if is_prod else "training", *common,
                        folds_done, False, "v5: предыдущий срез ещё идёт")

    with open(LOCK, "w", encoding="utf-8") as f:
        f.write(str(time.time()))
    out = os.path.join(MIX, "v5_prod.json" if is_prod else "v5_cv.json")
    cmd = [PY, os.path.join(HF, "GestureLab_PC_v0.1", "tools",
                            "finetune_ctc_phrases.py"),
           "--inventory", os.path.join(MIX, "v5_mix_inventory.json"),
           "--cache", os.path.join(MIX, "landmarks"),
           "--init", os.path.join(HF, "rsl_translator_v1", "prod_v4.pt"),
           "--work", WORK_PROD if is_prod else WORK,
           "--out", out,
           "--no-canon", "--aug", "hard", "--epochs", "30",
           "--recombine-frac", "0.25",
           "--align-from", "D:/hfdata/phrase_combined_dup/fold{fold}.pt",
           "--dup-times", "8",
           "--dup-ids", os.path.join(HF, "combined_phrases", "rutube_ids.json"),
           "--resume", "--no-llm"]
    if is_prod:
        cmd.append("--train-all")
    with open(os.path.join(MIX, "train.log"), "ab") as logf:
        try:
            proc = subprocess.run(cmd, cwd=os.path.join(HF, "GestureLab_PC_v0.1"),
                                  env=env, stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=TRAIN_BUDGET_S)
        except subprocess.TimeoutExpired:
            proc = None
    if proc is not None and proc.returncode != 0:
        write_status(phase="error", cues=cues, done_videos=done_v,
                     eligible_videos=eligible, folds_done=folds_done,
                     returncode=proc.returncode)
        raise RuntimeError(f"Обучение v5 завершилось с кодом {proc.returncode}; см. v5_mix/train.log")

    folds_done = progress_folds()
    rep_prod = prod_report()
    if rep_prod:
        digest = publish_prod(rep_prod)
        write_status(phase="done", cues=cues, done_videos=done_v,
                     eligible_videos=eligible, folds_done=FOLDS_TOTAL,
                     prod_sha256=digest)
        return artifact("done", *common, FOLDS_TOTAL, True,
                        "v5 ГОТОВА: CV 5/5, prod_v5.pt опубликован",
                        prod_sha256=digest)
    if is_prod:
        write_status(phase="prod", cues=cues, done_videos=done_v,
                     eligible_videos=eligible, folds_done=FOLDS_TOTAL)
        return artifact("prod", *common, FOLDS_TOTAL, False,
                        "v5: production train-all продолжится в следующем срезе")
    if done_report_exists():
        write_status(phase="cv_done", cues=cues, done_videos=done_v,
                     eligible_videos=eligible, folds_done=FOLDS_TOTAL)
        return artifact("cv_done", *common, FOLDS_TOTAL, False,
                        "v5: CV 5/5 готово; следующий срез — production")
    write_status(phase="training", cues=cues, done_videos=done_v,
                 eligible_videos=eligible, folds_done=folds_done)
    return artifact("training", *common, folds_done, False,
                    f"v5: обучение идёт, фолды {folds_done}/5, корпус {cues}")
