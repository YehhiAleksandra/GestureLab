"""Deterministic signer-disjoint split for Slovo landmarks.

Read-only w.r.t. the dataset: writes only the split/freeze JSON files.

Rule (v1):
- Splits are assigned per signer; no signer appears in more than one split.
- Signers are visited in descending clip count (ties by id) and assigned to the
  split with the largest remaining clip deficit, so clip shares approach
  train/val/test targets while respecting the heavy skew of Slovo signers.
- The test split is frozen: its signer list, clip ids and a fingerprint are
  written once and never used for model selection.

Usage:
  python tools/split_signer_disjoint.py \
      --data datasets/slovo/slovo_holistic.npz \
      --out <dir> --seed 42 --test-frac 0.20 --val-frac 0.10
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "slovo_signer_disjoint_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(data: Path):
    d = np.load(data)
    ids = [x.decode() if isinstance(x, bytes) else str(x) for x in d["ids"]]
    texts = [x.decode() if isinstance(x, bytes) else str(x) for x in d["texts"]]
    users = [x.decode() if isinstance(x, bytes) else str(x) for x in d["users"]]
    assert len(ids) == len(texts) == len(users), "npz arrays must align"
    assert len(set(ids)) == len(ids), "clip ids must be unique"
    return ids, texts, users


def assign(users: list[str], fracs: dict[str, float], seed: int) -> dict[str, str]:
    counts = Counter(users)
    total = sum(counts.values())
    targets = {k: total * v for k, v in fracs.items()}
    assigned = {k: 0 for k in fracs}
    # deterministic order: clip count desc, then id; seed only breaks ties by id
    order = sorted(counts, key=lambda s: (-counts[s], s))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)  # tie-shuffle, then re-sort stably by size
    order.sort(key=lambda s: -counts[s])
    out: dict[str, str] = {}
    for signer in order:
        # choose split with largest remaining deficit; tie -> fixed priority
        best = max(fracs, key=lambda k: (targets[k] - assigned[k], k == "train"))
        out[signer] = best
        assigned[best] += counts[signer]
    return out


def coverage(ids, texts, users, split_of, split):
    idx = [i for i, u in enumerate(users) if split_of[u] == split]
    per_class = Counter(texts[i] for i in idx)
    return {
        "clips": len(idx),
        "signers": len({users[i] for i in idx}),
        "classes": len(per_class),
        "no_event_clips": per_class.get("no_event", 0),
        "clip_ids": [ids[i] for i in idx],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/slovo/slovo_holistic.npz")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--val-frac", type=float, default=0.10)
    a = ap.parse_args()

    data = Path(a.data)
    assert abs(a.test_frac + a.val_frac) < 1, "train fraction must stay positive"
    fracs = {"train": 1.0 - a.test_frac - a.val_frac, "val": a.val_frac, "test": a.test_frac}

    ids, texts, users = load(data)
    split_of = assign(users, fracs, a.seed)

    # hard invariants
    for s in set(split_of.values()):
        assert any(split_of[u] == s for u in users), f"empty split {s}"
    assert len({u for u in users if split_of[u] == "test"}) > 0
    assert all(split_of[u] in fracs for u in users)

    stats = {split: coverage(ids, texts, users, split_of, split) for split in fracs}
    all_classes = {t for t in texts}
    missing_train = sorted(all_classes - {texts[i] for i, u in enumerate(users) if split_of[u] == "train"})

    # fingerprint: dataset content + exact clip->split assignment
    lines = sorted(f"{ids[i]}\t{users[i]}\t{split_of[users[i]]}\t{texts[i]}" for i in range(len(ids)))
    body = "\n".join(lines)
    dataset_sha = sha256_file(data)
    fp = hashlib.sha256((dataset_sha + "\n" + body).encode()).hexdigest()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = len(ids)
    split_doc = {
        "version": VERSION,
        "dataset": str(data),
        "dataset_sha256": dataset_sha,
        "seed": a.seed,
        "rule": "signer-disjoint; signers sorted by clip count desc; greedy largest-deficit assignment",
        "targets": fracs,
        "achieved_clip_share": {k: round(v["clips"] / total, 4) for k, v in stats.items()},
        "counts": {k: {"clips": v["clips"], "signers": v["signers"], "classes": v["classes"],
                       "no_event_clips": v["no_event_clips"]} for k, v in stats.items()},
        "class_coverage": {k: round(v["classes"] / len(all_classes), 4) for k, v in stats.items()},
        "classes_missing_in_train": missing_train,
        "fingerprint_sha256": fp,
        "signer_split": dict(sorted(split_of.items())),
        "clip_index": {s: [i for i, u in enumerate(users) if split_of[u] == s] for s in fracs},
    }
    (out / f"{VERSION}.json").write_text(json.dumps(split_doc, ensure_ascii=False, indent=1), encoding="utf-8")

    test_ids = stats["test"]["clip_ids"]
    freeze = {
        "version": VERSION,
        "sealed": True,
        "dataset_sha256": dataset_sha,
        "split_fingerprint_sha256": fp,
        "test_signers": sorted(u for u in set(users) if split_of[u] == "test"),
        "test_clip_ids": sorted(test_ids),
        "test_clip_ids_sha256": hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest(),
        "n_clips": len(test_ids),
        "n_signers": stats["test"]["signers"],
        "classes_present": stats["test"]["classes"],
        "policy": "test is evaluated once at final measurement; never used for checkpoint selection",
    }
    (out / f"{VERSION}.test_freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = {
        "version": VERSION,
        "fingerprint": fp,
        "clip_share": split_doc["achieved_clip_share"],
        "counts": split_doc["counts"],
        "class_coverage": split_doc["class_coverage"],
        "classes_missing_in_train": len(missing_train),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
