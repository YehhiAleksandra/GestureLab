"""Iteration 2: SignNet (holistic landmarks) on the sealed signer-disjoint split v1.

Contract (coordinator, 2026-10-28):
- train on split v1 only (splits/slovo_signer_disjoint_v1.json, clip_index);
- 3 seeds x 40 epochs on the customer GPU;
- checkpoint selection is val-only: best val top-1, tiebreak val macro-F1;
- the sealed test split is read and scored exactly once, after the best seed is chosen;
- outputs: predictions_test.csv + eval_test.json (with an `acceptance` block).

Integrity gates before training:
- dataset sha256 equals the one recorded in the split/freeze;
- test clip ids hash equals slovo_signer_disjoint_v1.test_freeze.json.

Usage:
  py -3 tools/train_slovo_split.py --out-dir runs/iter2
  py -3 tools/train_slovo_split.py --out-dir runs/iter2_smoke --seeds 1 --epochs 2 \
      --limit-train 2048 --limit-val 256 --skip-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slovo_model import HFEAT, SignNet, T, augment, holistic_features, resample  # noqa: E402

PRIMARY = Path(r"D:/ДляГлухихAi/GestureLab_PC_v0.1")
DATASET_VERSION = "slovo_holistic_v1+slovo_signer_disjoint_v1"
SUBSET_VERSION = "+test-subset"  # only set for smoke runs, never for the official measurement

ACCEPTANCE = {"top1": 0.95, "macro_f1": 0.93, "p95_latency_ms": 300.0}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path):
    d = np.load(path)
    ids = [str(x) for x in d["ids"]]
    texts = [str(x) for x in d["texts"]]
    users = [str(x) for x in d["users"]]
    return d["offsets"], d["frames"], ids, texts, users


def build_resampled(frames, offsets, n: int) -> np.ndarray:
    R = np.empty((n, T, frames.shape[1], frames.shape[2]), np.float16)
    t0 = time.time()
    for i in range(n):
        R[i] = resample(frames[offsets[i]:offsets[i + 1]], T)
    print(f"resampled cache {R.shape} {R.nbytes / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    return R


def feats_aug(R: np.ndarray, idx: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng([seed, epoch, int(idx)])
    return holistic_features(augment(R[idx].astype(np.float32), rng))


def feats_plain(R: np.ndarray, idx: int) -> np.ndarray:
    return holistic_features(R[idx].astype(np.float32))


@torch.no_grad()
def predict(model, X: np.ndarray, dev: str, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)


def topk_acc(logits: np.ndarray, y: np.ndarray, k: int) -> float:
    idx = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    return float((idx == y[:, None]).any(1).mean())


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def cluster_bootstrap(y, pred, signers, n_boot: int, rng):
    uniq = np.unique(signers)
    per = {s: np.where(signers == s)[0] for s in uniq}
    t1, f1 = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([per[s] for s in pick])
        t1.append(float((pred[idx] == y[idx]).mean()))
        f1.append(float(f1_score(y[idx], pred[idx], average="macro", zero_division=0)))
    return ([round(float(v), 4) for v in np.percentile(t1, [2.5, 97.5])],
            [round(float(v), 4) for v in np.percentile(f1, [2.5, 97.5])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PRIMARY / "datasets/slovo/slovo_holistic.npz"))
    ap.add_argument("--split", default="splits/slovo_signer_disjoint_v1.json")
    ap.add_argument("--freeze", default="splits/slovo_signer_disjoint_v1.test_freeze.json")
    ap.add_argument("--out-dir", default="runs/iter2")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--skip-test", action="store_true", help="smoke only: do not touch the sealed test")
    ap.add_argument("--model-version", default="signnet_holistic_iter2")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    torch.backends.cudnn.benchmark = True

    # ---------- data + split integrity ----------
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    freeze = json.loads(Path(a.freeze).read_text(encoding="utf-8"))
    offsets, frames, ids, texts, users = load_npz(Path(a.data))
    n = len(ids)
    assert n == len(offsets) - 1 == len(texts) == len(users), "npz arrays must align"
    assert freeze["n_clips"] == len(freeze["test_clip_ids"]), "freeze file is inconsistent"
    assert freeze["split_fingerprint_sha256"] == split["fingerprint_sha256"], "split/freeze mismatch"

    idx = {s: np.asarray(split["clip_index"][s], dtype=np.int64) for s in ("train", "val", "test")}
    test_ids_sorted = sorted(ids[i] for i in idx["test"])
    test_hash = hashlib.sha256("\n".join(test_ids_sorted).encode()).hexdigest()
    assert test_hash == freeze["test_clip_ids_sha256"], "test clip ids do not match the sealed freeze"
    print(f"split v1 ok: {split['fingerprint_sha256'][:12]}… test_freeze {test_hash[:12]}… "
          f"train/val/test = {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}", flush=True)

    names = sorted(set(texts))
    name_to_idx = {t: i for i, t in enumerate(names)}
    y = np.asarray([name_to_idx[t] for t in texts], dtype=np.int64)
    assert not split["classes_missing_in_train"], "train split must cover every class"

    tr = idx["train"][: a.limit_train or None]
    va = idx["val"][: a.limit_val or None]
    te = idx["test"][: a.limit_test or None]

    R = build_resampled(frames, offsets, n)
    del offsets, frames

    t0 = time.time()
    Xv = np.stack([feats_plain(R, i) for i in va])
    Xt = None if a.skip_test else np.stack([feats_plain(R, i) for i in te])
    print(f"features val {Xv.shape} / test {'none' if Xt is None else Xt.shape} "
          f"in {time.time() - t0:.0f}s", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}", flush=True)
    if dev == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)

    pool = ThreadPoolExecutor(max_workers=a.workers)
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SignNet(len(names), feat=HFEAT).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        nb = len(tr) // a.bs
        assert nb > 0, "train split smaller than batch size"
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)
        rng = np.random.default_rng(seed)
        sd = out / f"seed{seed}"
        sd.mkdir(parents=True, exist_ok=True)

        best = (-1.0, -1.0)
        best_meta = {}
        history = []
        t_seed = time.time()
        for ep in range(1, a.epochs + 1):
            model.train()
            perm = rng.permutation(tr)
            total = 0.0
            t_ep = time.time()
            for b in range(nb):
                batch = perm[b * a.bs:(b + 1) * a.bs]
                xb = torch.from_numpy(np.stack(list(pool.map(lambda i: feats_aug(R, i, seed, ep), batch)))).to(dev)
                yb = torch.as_tensor(y[batch], device=dev)
                loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += float(loss.item())

            logits = predict(model, Xv, dev)
            v_top1, v_top5 = topk_acc(logits, y[va], 1), topk_acc(logits, y[va], 5)
            v_f1 = macro_f1(y[va], logits.argmax(1))
            improved = (v_top1, v_f1) > best
            if improved:
                best = (v_top1, v_f1)
                best_meta = {"epoch": ep, "val_top1": round(v_top1, 4), "val_top5": round(v_top5, 4),
                             "val_macro_f1": round(v_f1, 4)}
                torch.save({"model": model.state_dict(), "names": names, "kind": "holistic",
                            "feat": HFEAT, "seed": seed, **best_meta},
                           sd / "best.pt")
            history.append({"epoch": ep, "train_loss": round(total / nb, 4),
                            "val_top1": round(v_top1, 4), "val_top5": round(v_top5, 4),
                            "val_macro_f1": round(v_f1, 4), "best": bool(improved),
                            "lr": round(sched.get_last_lr()[0], 8), "sec": round(time.time() - t_ep, 1)})
            (sd / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
            print(f"seed {seed} ep {ep:02d}/{a.epochs} loss {total / nb:.3f} "
                  f"val top1 {v_top1:.4f} f1 {v_f1:.4f} best {best[0]:.4f}/{best[1]:.4f}"
                  f"{' *' if improved else ''} [{time.time() - t_ep:.0f}s, seed {time.time() - t_seed:.0f}s]",
                  flush=True)

        runs.append({"seed": seed, "best": best_meta, "seconds": round(time.time() - t_seed, 1),
                     "checkpoint": str(sd / "best.pt")})
        print(f"seed {seed} done: {best_meta} in {time.time() - t_seed:.0f}s", flush=True)

    chosen = max(runs, key=lambda r: (r["best"]["val_top1"], r["best"]["val_macro_f1"], -r["seed"]))
    print(f"chosen by val: seed {chosen['seed']} {chosen['best']}", flush=True)

    summary = {
        "dataset_version": DATASET_VERSION + (SUBSET_VERSION if a.skip_test else ""),
        "split_fingerprint_sha256": split["fingerprint_sha256"],
        "test_clip_ids_sha256": test_hash,
        "dataset_sha256": split["dataset_sha256"],
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "n_classes": len(names), "epochs": a.epochs, "batch_size": a.bs, "lr": a.lr,
        "checkpoint_selection": {
            "policy": "val-only: best val top-1 per seed, tiebreak val macro-F1; test never used",
            "metric": "val_top1", "tiebreak": "val_macro_f1",
            "per_seed": runs,
            "chosen_seed": chosen["seed"],
            "chosen_val": chosen["best"],
        },
    }
    (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    pool.shutdown()

    if a.skip_test:
        print(json.dumps({"smoke": True, "summary": summary["checkpoint_selection"]}, ensure_ascii=False))
        return

    # ---------- single sealed-test evaluation ----------
    ck = torch.load(chosen["checkpoint"], map_location=dev, weights_only=False)
    model = SignNet(len(names), feat=HFEAT).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    logits = predict(model, Xt, dev)
    pred = logits.argmax(1)
    y_te = y[te]
    signers = np.asarray([users[i] for i in te])
    rng = np.random.default_rng(20261028)
    ci_top1, ci_f1 = cluster_bootstrap(y_te, pred, signers, a.bootstrap, rng)

    # per-clip latency: features -> forward, bs=1, CUDA synchronised
    for i in te[:20]:
        model(torch.from_numpy(feats_plain(R, int(i))[None]).to(dev))
    if dev == "cuda":
        torch.cuda.synchronize()
    lat = []
    for i in te:
        t = time.perf_counter()
        x = torch.from_numpy(feats_plain(R, int(i))[None]).to(dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        with torch.no_grad():
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000.0)
    lat = np.asarray(lat)

    top1 = float((pred.numpy() == y_te).mean())
    top5 = topk_acc(logits, y_te, 5)
    f1m = macro_f1(y_te, pred.numpy())
    p50, p95, p99 = (float(np.percentile(lat, q)) for q in (50, 95, 99))
    hits = int((pred.numpy() == y_te).sum())

    version = f"{a.model_version}-seed{chosen['seed']}"
    order = np.argsort(-logits, axis=1)[:, :5]
    pred_path = out / "predictions_test.csv"
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        f.write("sample_id,rank1,rank2,rank3,rank4,rank5,latency_ms,model_version\n")
        for row, i in enumerate(te):
            ranks = [names[int(c)] for c in order[row]]
            f.write(f"{ids[int(i)]}," + ",".join(f'"{r}"' for r in ranks) +
                    f",{lat[row]:.3f},{version}\n")

    passed = {"top1": top1 >= ACCEPTANCE["top1"], "macro_f1": f1m >= ACCEPTANCE["macro_f1"],
              "p95_latency_ms": p95 <= ACCEPTANCE["p95_latency_ms"]}
    eval_doc = {
        "dataset_version": DATASET_VERSION,
        "model_version": version,
        "kind": "holistic",
        "split_fingerprint_sha256": split["fingerprint_sha256"],
        "test_clip_ids_sha256": test_hash,
        "dataset_sha256": split["dataset_sha256"],
        "checkpoint_selection": summary["checkpoint_selection"],
        "selection_confirmation": (
            "Checkpoint selection used val only (val top-1, tiebreak val macro-F1); "
            "the sealed test split was scored exactly once for this report."),
        "test": {
            "n_clips": int(len(te)), "n_signers": int(len(np.unique(signers))),
            "n_classes_present": int(len(np.unique(y_te))),
            "top1": round(top1, 4), "top5": round(top5, 4), "hits_top1": hits,
            "macro_f1": round(f1m, 4),
            "top1_wilson95": wilson_ci(hits, len(te)),
            "top1_cluster_bootstrap95": ci_top1,
            "macro_f1_cluster_bootstrap95": ci_f1,
        },
        "latency": {
            "n": int(len(lat)), "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3), "mean_ms": round(float(lat.mean()), 3),
            "measured": "holistic_features -> GPU forward, batch=1, CUDA synchronised",
            "hardware": "NVIDIA GeForce RTX 3060 Ti 8 GB (customer PC)",
            "note": "camera capture/MediaPipe tracking latency is measured separately, not included",
        },
        "acceptance": {
            "targets": ACCEPTANCE,
            "measured": {"top1": round(top1, 4), "macro_f1": round(f1m, 4),
                         "p95_latency_ms": round(p95, 3)},
            "passed": passed,
            "overall_pass": bool(all(passed.values())),
            "protected_delta_pp": {
                "target_max_pp": 2.0, "status": "not_evaluated",
                "reason": "no protected class list in the v1 contract"},
        },
        "training": {"epochs": a.epochs, "batch_size": a.bs, "lr": a.lr,
                     "seeds": seeds, "n_train": int(len(tr)), "n_val": int(len(va))},
        "notes": [
            "Augmentation: mirror/rotation/scale/shift/noise on T=32 resampled landmarks; random temporal crop omitted",
            "Test split read once, after seed selection by val",
            "dataset_version contains no 'official' marker",
        ],
    }
    (out / "eval_test.json").write_text(json.dumps(eval_doc, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps({"model_version": version, "test": eval_doc["test"],
                      "latency": eval_doc["latency"], "acceptance": eval_doc["acceptance"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
