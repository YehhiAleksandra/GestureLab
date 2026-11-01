"""Legacy signer-independent protocol behind the 61.0% top-1 baseline.

Legacy definition (from reports/baseline_20261027.md):
  test  = clips with official split flag train == False
  unseen = test clips whose signer does not occur in the official train set
No separate val split existed; the checkpoint was selected on the official test.

This tool re-runs the production SignNet on those clips and reports top-1,
top-5, macro-F1 together with (a) Wilson CI over clips and (b) signer-cluster
bootstrap CI. Read-only w.r.t. the dataset.

Usage:
  python tools/protocol_legacy_61.py --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slovo_model import KINDS, SignNet, T, holistic_features, resample  # noqa: E402

DEFAULT_DATA = Path(r"D:/ДляГлухихAi/GestureLab_PC_v0.1/datasets/slovo/slovo_holistic.npz")
DEFAULT_CKPT = Path(r"D:/ДляГлухихAi/GestureLab_PC_v0.1/models/slovo_signnet_holistic.pt")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", default=".")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    DATA, CKPT = a.data, a.ckpt

    d = np.load(DATA)
    o, frames, ids = d["offsets"], d["frames"], d["ids"]
    ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]
    texts = [x.decode() if isinstance(x, bytes) else str(x) for x in d["texts"]]
    users = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in d["users"]])
    train = d["train"].astype(bool)
    names = sorted(set(texts))
    y = np.array([names.index(t) for t in texts])
    seqs = [frames[o[i]:o[i + 1]] for i in range(len(o) - 1)]

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = SignNet(len(ck["names"]), feat=KINDS[ck["kind"]][1])
    model.load_state_dict(ck["model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    X = torch.from_numpy(np.stack([holistic_features(resample(s, T)) for s in seqs]))
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 512].to(dev)).float().cpu() for i in range(0, len(X), 512)])
    probs = logits.softmax(1).numpy()
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, 1)[:, :5]

    te = np.where(~train)[0]
    tr_users = set(users[train].tolist())
    unseen = np.array([i for i in te if users[i] not in tr_users])

    correct = pred[unseen] == y[unseen]
    top1 = float(correct.mean())
    n = len(unseen)
    z = 1.96
    den = 1 + z * z / n
    ctr = (top1 + z * z / (2 * n)) / den
    hw = z * np.sqrt(top1 * (1 - top1) / n + z * z / (4 * n * n)) / den
    wilson = [round(ctr - hw, 4), round(ctr + hw, 4)]

    # signer-cluster bootstrap
    signers = sorted(set(users[unseen].tolist()))
    by_signer = {s: unseen[users[unseen] == s] for s in signers}
    rng = np.random.default_rng(0)
    b_top1, b_f1 = [], []
    for _ in range(a.bootstrap):
        pick = rng.choice(len(signers), len(signers), replace=True)
        idx = np.concatenate([by_signer[signers[k]] for k in pick])
        b_top1.append(float((pred[idx] == y[idx]).mean()))
        b_f1.append(f1_score(y[idx], pred[idx], average="macro", zero_division=0))
    cluster_ci = {
        "top1": [round(float(v), 4) for v in np.percentile(b_top1, [2.5, 97.5])],
        "macro_f1": [round(float(v), 4) for v in np.percentile(b_f1, [2.5, 97.5])],
    }

    per_sign = {}
    for s in signers:
        ix = by_signer[s]
        per_sign[s] = {"clips": int(len(ix)), "top1": round(float((pred[ix] == y[ix]).mean()), 4)}
    accs = np.array([v["top1"] for v in per_sign.values()])

    doc = {
        "protocol": "legacy-official-test-unseen-signers",
        "dataset": str(DATA),
        "dataset_sha256": sha256_file(DATA),
        "checkpoint": str(CKPT),
        "n_test_official": int(len(te)),
        "n_unseen_clips": int(n),
        "n_unseen_signers": len(signers),
        "unseen_signers": signers,
        "unseen_clip_ids_sha256": hashlib.sha256("\n".join(sorted(ids[i] for i in unseen)).encode()).hexdigest(),
        "metrics": {
            "top1": round(top1, 4),
            "top1_hits": int(correct.sum()),
            "top5": round(float((top5[unseen] == y[unseen][:, None]).any(1).mean()), 4),
            "macro_f1": round(float(f1_score(y[unseen], pred[unseen], average="macro", zero_division=0)), 4),
            "classes_present": len(set(y[unseen].tolist())),
        },
        "ci": {
            "wilson_clips_95": wilson,
            "cluster_bootstrap_signers_95": cluster_ci,
            "note": "Wilson treats clips as independent; cluster bootstrap resamples signers and is the authoritative CI",
        },
        "per_signer": per_sign,
        "per_signer_top1_mean": round(float(accs.mean()), 4),
        "per_signer_top1_median": round(float(np.median(accs)), 4),
        "per_signer_top1_min": round(float(accs.min()), 4),
        "per_signer_top1_max": round(float(accs.max()), 4),
        "per_signer_top1_p10_p90": [round(float(v), 4) for v in np.percentile(accs, [10, 90])],
        "predictions": {ids[i]: {"signer": users[i], "true": texts[i], "pred": names[int(pred[i])]} for i in unseen},
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "legacy_61_protocol.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ["n_unseen_clips", "n_unseen_signers", "metrics", "ci",
                                          "per_signer_top1_mean", "per_signer_top1_p10_p90"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
