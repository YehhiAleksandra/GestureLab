# -*- coding: utf-8 -*-
"""v4 trainer: автообучение CTC v4, когда корпус ВОГinfo достиг 10000 cue.

Фазы:
  waiting  — cues < 10000: дешёвый artifact, ничего не делаем
  snapshot — один раз: инвентарь ytsl25 + мерж с 1250 фразами (жёсткие ссылки)
  training — срез обучения (--resume, докачка по фолдам/эпохам), бюджет ~55 мин
  done     — progress.json имеет все 5 фолдов, v4_cv.json без partial

Статус дублируется в D:\\hfdata\\v4_status.json (читает нотифаер).
"""
import glob
import json
import os
import subprocess
import time

PY = r"D:\venvs\gesturelab\Scripts\python.exe"
HF = r"D:\hfdata"
CUES = os.path.join(HF, "ytsl25_rsl", "cues")
MIX = os.path.join(HF, "v4_mix")
WORK = os.path.join(MIX, "work")
STATUS = os.path.join(HF, "v4_status.json")
NEEDED = 10000
FOLDS_TOTAL = 5
TRAIN_BUDGET_S = 3300
LOCK = os.path.join(MIX, "trainer.lock")


def count_cues():
    n, done_v = 0, 0
    for f in glob.glob(os.path.join(CUES, "*.blocks.json")):
        try:
            st = json.loads(open(f, encoding="utf-8").read())
        except Exception:
            continue
        if st.get("done"):
            done_v += 1
            n += len(st.get("meta", []))
    return n, done_v


def write_status(**kw):
    base = {}
    if os.path.exists(STATUS):
        try:
            base = json.loads(open(STATUS, encoding="utf-8").read())
        except Exception:
            base = {}
    base.update(kw)
    base["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    open(STATUS, "w", encoding="utf-8").write(json.dumps(base, ensure_ascii=False))


def progress_folds():
    p = os.path.join(WORK, "progress.json")
    if not os.path.exists(p):
        return 0
    try:
        return len(json.loads(open(p, encoding="utf-8").read()).get("folds", {}))
    except Exception:
        return 0


def done_report_exists():
    out = os.path.join(MIX, "v4_cv.json")
    if not os.path.exists(out):
        return False
    try:
        rep = json.loads(open(out, encoding="utf-8").read())
        return not rep.get("partial") and "held_beam" in rep
    except Exception:
        return False


def prod_done():
    p = os.path.join(MIX, "v4_prod.json")
    if not os.path.exists(p):
        return False
    try:
        return bool(json.loads(open(p, encoding="utf-8").read()).get("prod"))
    except Exception:
        return False


def artifact(phase, cues, folds_done, complete, summary):
    return {"artifact": {"summary": summary, "phase": phase, "cues": cues,
                         "folds_done": folds_done, "folds_total": FOLDS_TOTAL,
                         "complete": complete}}


def run(ctx):
    cues, done_v = count_cues()
    folds_done = progress_folds()

    if done_report_exists() and prod_done():
        write_status(phase="done", cues=cues, folds_done=FOLDS_TOTAL)
        return artifact("done", cues, FOLDS_TOTAL, True,
                        "v4 ГОТОВА: CV 5/5 + prod v4_mix/v4_prod.json")

    is_prod = done_report_exists()  # CV завершён → фаза prod train-all

    if cues < NEEDED:
        write_status(phase="waiting", cues=cues, folds_done=folds_done)
        return artifact("waiting", cues, folds_done, False,
                        f"v4: ждём корпус {cues}/{NEEDED} cue-фрагментов "
                        f"(видео обработано полностью: {done_v})")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    os.makedirs(os.path.join(MIX, "landmarks"), exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    marker = os.path.join(MIX, "snapshot.ok")
    if not os.path.exists(marker):
        subprocess.run([PY, os.path.join(HF, "ytsl25_inventory.py")],
                         cwd=HF, env=env, capture_output=True, timeout=600)
        a = json.load(open(os.path.join(HF, "combined_phrases",
                                        "combined_inventory.json"), encoding="utf-8"))
        b = json.load(open(os.path.join(HF, "ytsl25_rsl",
                                        "ytsl25_inventory.json"), encoding="utf-8"))
        linked = 0
        for src_dir, inv in [(os.path.join(HF, "combined_phrases", "landmarks"), a),
                             (CUES, b)]:
            for p_ in inv["phrases"]:
                s = os.path.join(src_dir, p_["id"] + ".npz")
                d = os.path.join(MIX, "landmarks", p_["id"] + ".npz")
                if os.path.exists(s) and not os.path.exists(d):
                    try:
                        os.link(s, d)
                        linked += 1
                    except OSError:
                        pass
        merged = {"channel": "v4: combined_phrases(1250) + ytsl25_rsl VOGinfo cues",
                  "phrases": a["phrases"] + b["phrases"]}
        open(os.path.join(MIX, "v4_mix_inventory.json"), "w",
             encoding="utf-8").write(json.dumps(merged, ensure_ascii=False))
        open(marker, "w", encoding="utf-8").write(str(linked))

    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 3500:
        return artifact("prod" if is_prod else "training", cues, folds_done, False,
                        f"v4: срез ещё идёт (лок), фолды {folds_done}/5")

    work_prod = os.path.join(MIX, "work_prod")
    if is_prod:
        os.makedirs(work_prod, exist_ok=True)
    open(LOCK, "w").write(str(time.time()))
    out = os.path.join(MIX, "v4_prod.json" if is_prod else "v4_cv.json")
    cmd = [PY, os.path.join(HF, "GestureLab_PC_v0.1", "tools",
                            "finetune_ctc_phrases.py"),
           "--inventory", os.path.join(MIX, "v4_mix_inventory.json"),
           "--cache", os.path.join(MIX, "landmarks"),
           "--init", os.path.join(HF, "rsl_translator_v1", "prod_v3.pt"),
           "--work", work_prod if is_prod else WORK,
           "--out", out,
           "--no-canon", "--aug", "hard", "--epochs", "30",
           "--recombine-frac", "0.25",
           "--align-from", "D:/hfdata/phrase_combined_dup/fold{fold}.pt",
           "--dup-times", "8",
           "--dup-ids", os.path.join(HF, "combined_phrases", "rutube_ids.json"),
           "--resume", "--no-llm"]
    if is_prod:
        cmd.append("--train-all")
    logf = open(os.path.join(MIX, "train.log"), "ab")
    try:
        subprocess.run(cmd, cwd=os.path.join(HF, "GestureLab_PC_v0.1"),
                       env=env, stdout=logf, stderr=subprocess.STDOUT,
                       timeout=TRAIN_BUDGET_S)
    except subprocess.TimeoutExpired:
        pass
    finally:
        logf.close()

    folds_done = progress_folds()
    if is_prod:
        if prod_done():
            write_status(phase="done", cues=cues, folds_done=FOLDS_TOTAL)
            return artifact("done", cues, FOLDS_TOTAL, True,
                            "v4 ГОТОВА: CV 5/5 + prod v4_mix/v4_prod.json")
        write_status(phase="prod", cues=cues, folds_done=FOLDS_TOTAL)
        return artifact("prod", cues, FOLDS_TOTAL, False,
                        "v4: prod train-all на всех 11405 клипах, идёт")
    if done_report_exists():
        write_status(phase="done", cues=cues, folds_done=FOLDS_TOTAL)
        return artifact("done", cues, FOLDS_TOTAL, True, "v4 ОБУЧЕНА: фолды 5/5")
    write_status(phase="training", cues=cues, folds_done=folds_done)
    return artifact("training", cues, folds_done, False,
                    f"v4: обучение идёт, фолды готовы {folds_done}/5, корпус {cues}")
