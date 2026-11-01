# -*- coding: utf-8 -*-
"""Скачивание RSL-видео SP-10 (ru-переводы) с прогрессом и докачкой."""
import json, os, time, urllib.request, sys

OUT = r"D:\hfdata\sp10_rsl\videos"
os.makedirs(OUT, exist_ok=True)
STATE = r"D:\hfdata\sp10_rsl\download_state.json"

state = {}
if os.path.exists(STATE):
    state = json.load(open(STATE, encoding="utf-8"))

entries = []  # (uid, text, url)
for split in ("train", "dev", "test"):
    d = json.load(open(rf"D:\hfdata\sp10_repo\dataset\{split}.json", encoding="utf-8"))
    for e in d:
        ru = [s for s in e["sign_list"] if s["lang"] == "ru"]
        if not ru:
            continue
        uid = f"{split}_{e['id']}"
        entries.append((uid, ru[0]["text"], ru[0]["video_url"]))

todo = [x for x in entries if state.get(x[0], {}).get("ok")]
missing = [x for x in entries if not state.get(x[0], {}).get("ok")]
print(f"total={len(entries)} ok={len(todo)} missing={len(missing)}")

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
t0 = time.time()
done = fail = 0
for uid, text, url in missing:
    if time.time() - t0 > BUDGET:
        break
    path = os.path.join(OUT, uid + ".mp4")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r, open(path, "wb") as f:
            f.write(r.read())
        state[uid] = {"ok": True, "text": text, "url": url, "size": os.path.getsize(path)}
        done += 1
    except Exception as ex:
        state[uid] = {"ok": False, "text": text, "url": url, "err": str(ex)[:120]}
        fail += 1
    if (done + fail) % 50 == 0:
        json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)

json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)
ok = sum(1 for v in state.values() if v.get("ok"))
print(f"session done={done} fail={fail} total_ok={ok}/{len(entries)}")
