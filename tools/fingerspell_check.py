"""Проверка дактиля в holy_sepulchre: ловит ли модель букв имена, которые звучат в речи.

Модель не видит эталон: буквы предсказываются только по ландмаркам,
речь используется после предсказания, как «ответы».
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_pytorch import GestureGRU  # noqa: E402

VIDEO = "datasets/youtube_ark/holy_sepulchre_landmarks.json"
MODEL = "bukva_letters.pt"
REF = "reports/ref/holy_sepulchre.json"
WINDOW, STEP = 22, 5
MIN_CONF = 0.5
NAMES = ["ХРИСТОС", "ГОЛГОФА", "АДАМ", "МАРИЯ", "ИИСУС", "ДАВИД", "ХРАМ", "КРЕСТ", "ГРОБ", "МОГИЛА"]
LAG = (0.3, 6.0)  # где искать буквы относительно начала слова в речи


def frame_features(frames: list[dict]) -> np.ndarray:
    """(N,126): рука1 + рука2 (нет второй — копия первой), нет рук — NaN."""
    out = np.full((len(frames), 126), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        if f["hands"]:
            h1 = np.asarray(f["hands"][0], dtype=np.float32).ravel()
            h2 = np.asarray(f["hands"][1] if len(f["hands"]) > 1 else f["hands"][0], dtype=np.float32).ravel()
            out[i] = np.concatenate([h1, h2])
    return out


def predict_letters(model, names: list[str], feats: np.ndarray, device) -> list[dict]:
    model.eval()
    idx_to_label = {i: str(n) for i, n in enumerate(names)}
    preds = []
    with torch.no_grad():
        for start in range(0, len(feats) - WINDOW + 1, STEP):
            w = feats[start:start + WINDOW]
            if np.isnan(w).any():
                continue
            x = torch.FloatTensor(w[None]).to(device)
            prob = torch.softmax(model(x), dim=1)[0]
            conf, pred = prob.max(0)
            preds.append({"frame": start + WINDOW // 2, "time": (start + WINDOW // 2) / 25.0,
                          "label": idx_to_label[pred.item()], "conf": conf.item()})
    return preds


def letter_stream(preds: list[dict]) -> list[dict]:
    """Сжимаем поток окон в сегменты букв: буква держится, пока та же буква в окнах."""
    segs = []
    for p in preds:
        if p["conf"] < MIN_CONF:
            continue
        if segs and segs[-1]["label"] == p["label"]:
            segs[-1]["end"] = p["time"]
            segs[-1]["n"] += 1
            segs[-1]["conf"].append(p["conf"])
        else:
            segs.append({"label": p["label"], "start": p["time"], "end": p["time"], "n": 1, "conf": [p["conf"]]})
    for s in segs:
        s["conf"] = float(np.mean(s["conf"]))
    return segs


def name_occurrences(ref: list[dict]) -> dict[str, list[float]]:
    out = {n: [] for n in NAMES}
    for seg in ref:
        for w in seg.get("words", []):
            for n in NAMES:
                if w["w"].upper().startswith(n):
                    out[n].append(w["s"])
    return out


def best_prefix_score(letters: str, name: str) -> int:
    """Сколько первых букв name набирается из letters по порядку (подпоследовательность)."""
    k = 0
    for ch in letters:
        if k < len(name) and ch == name[k]:
            k += 1
    return k


def check_names(segs: list[dict], occ: dict[str, list[float]]) -> None:
    letters = "".join(s["label"] for s in segs)
    t0 = np.array([s["start"] for s in segs])
    print(f"\n=== Сверка с речью (лаг {LAG[0]}–{LAG[1]} c) ===")
    for name, times in occ.items():
        if not times:
            continue
        hits, partial = [], []
        for t in times:
            lo, hi = np.searchsorted(t0, [t + LAG[0], t + LAG[1]])
            seg_letters = letters[lo:hi]
            score = best_prefix_score(seg_letters, name) if seg_letters else 0
            (hits if score >= len(name) else partial if score >= 2 else []).append((t, score, seg_letters))
        print(f"{name}: {len(hits)}/{len(times)} целиком, {len(partial)}/{len(times)} частично (>=2 букв)")
        for t, score, sl in (hits + partial)[:4]:
            print(f"    {t:.1f}с набрано {score}/{len(name)}: {''.join(sl)[:25]}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load("bukva_processed.npz", allow_pickle=True)
    ckpt = torch.load(MODEL, map_location=device, weights_only=False)
    model = GestureGRU(input_dim=126, hidden_dim=ckpt["hidden_dim"], num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    d = json.loads(Path(VIDEO).read_text(encoding="utf-8"))
    feats = frame_features(d["frames"])
    covered = np.isfinite(feats[:, 0]).mean()
    preds = predict_letters(model, list(data["names"]), feats, device)
    segs = letter_stream(preds)
    print(f"кадров: {len(feats)}, с руками: {covered:.0%}, окон: {len(preds)}, сегментов букв: {len(segs)}")
    cnt = Counter(s["label"] for s in segs)
    print("распределение букв:", ", ".join(f"{l}×{n}" for l, n in cnt.most_common(15)))
    dur = [(s["end"] - s["start"]) for s in segs if s["n"] > 1]
    if dur:
        print(f"медианная длительность повторяющейся буквы: {np.median(dur):.2f} c")

    ref = json.loads(Path(REF).read_text(encoding="utf-8"))
    check_names(segs, name_occurrences(ref))


if __name__ == "__main__":
    main()
