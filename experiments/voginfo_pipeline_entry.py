# -*- coding: utf-8 -*-
"""VOGinfo RSL pipeline: batch download + cue-landmark extraction.

Каждый запуск: до ~4 мин скачивания + до ~46 мин извлечения, затем прогресс.
Скрипты сами себя ограничивают бюджетом; state/кэш в D:\\hfdata\\ytsl25_rsl.
Каденция 1 час: суммарный бюджет (~50 мин + оверхед) меньше таймаута 60 мин.
Извлечение — долгий полюс, даём ему ~47 мин/час (~80% загрузки CPU).
"""
import json
import os
import subprocess
import sys

PY = r"D:\venvs\gesturelab\Scripts\python.exe"
HF = r"D:\hfdata"
VID = os.path.join(HF, "ytsl25_rsl", "videos")
CUES = os.path.join(HF, "ytsl25_rsl", "cues")


def run(ctx):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for argv, to in [([PY, os.path.join(HF, "ytsl25_dl.py"), "240"], 300),
                     ([PY, os.path.join(HF, "ytsl25_extract.py"), "3180"], 3240)]:
        try:
            subprocess.run(argv, cwd=HF, env=env, capture_output=True, timeout=to)
        except subprocess.TimeoutExpired:
            # бюджет перекрыт таймаутом (длинный блок): работа в кэше, идём дальше
            pass

    probe = json.load(open(os.path.join(HF, "ytsl25_rsl_probe.json"), encoding="utf-8"))
    ok_ids = [v for v, r in probe.items() if r.get("status") == "ok"]
    downloaded = sum(1 for v in ok_ids
                     if os.path.exists(os.path.join(VID, v + ".mp4"))
                     and os.path.getsize(os.path.join(VID, v + ".mp4")) > 100000)
    done_v = 0
    cues = 0
    if os.path.isdir(CUES):
        for f in os.listdir(CUES):
            if f.endswith(".blocks.json"):
                try:
                    st = json.loads(open(os.path.join(CUES, f), encoding="utf-8").read())
                    if st.get("done"):
                        done_v += 1
                        cues += len(st.get("meta", []))
                except Exception:
                    pass
    complete = downloaded >= len(ok_ids) and done_v >= downloaded and downloaded > 0
    summary = (f"VOGinfo: скачано {downloaded}/{len(ok_ids)} видео, "
               f"обработано {done_v}, cue-фрагментов {cues}"
               + (" — ВСЁ ГОТОВО" if complete else ""))
    return {"artifact": {"summary": summary, "videos_downloaded": downloaded,
                         "videos_total": len(ok_ids), "videos_done": done_v,
                         "cues": cues, "complete": complete}}
