#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Детальное обучение модели с логом по эпохам.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import normalize, GRU, topk


def train_model_detailed(X, y, is_val, epochs=100, hidden=128, batch=32, lr=0.01, seed=0, log_interval=10):
    """Обучает GRU с детальным логом по эпохам."""
    xn = normalize(X)
    model = GRU(xn.shape[2], hidden, int(y.max()) + 1, seed=seed)
    rng = np.random.default_rng(seed + 1)
    tr = ~is_val
    xtr, ytr = xn[tr], y[tr]
    xva, yva = xn[is_val], y[is_val]

    print(f"Датасет: {len(X)} окон, {int(y.max())+1} классов")
    print(f"Train: {len(xtr)}, Val: {len(xva)}")
    print(f"hidden={hidden}, epochs={epochs}, batch={batch}, lr={lr}")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Train top-1':>12} | {'Train top-3':>12} | {'Val top-1':>12} | {'Val top-3':>12}")
    print("-" * 70)

    step_i = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(xtr))
        for i in range(0, len(xtr), batch):
            idx = order[i:i + batch]
            prob, steps = model.forward(xtr[idx])
            model.step(model.backward(prob, ytr[idx], steps), lr, step_i := step_i + 1)

        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            ptr, _ = model.forward(xtr)
            pva, _ = model.forward(xva)
            t1 = topk(ptr, ytr, 1)
            t3 = topk(ptr, ytr, 3)
            v1 = topk(pva, yva, 1)
            v3 = topk(pva, yva, 3)
            print(f"{epoch:>6} | {t1:>11.1f}% | {t3:>11.1f}% | {v1:>11.1f}% | {v3:>11.1f}%")

    ptr, _ = model.forward(xtr)
    pva, _ = model.forward(xva)
    print("-" * 70)
    print(f"Финальные метрики:")
    print(f"  Train top-1: {topk(ptr, ytr, 1):.1f}%  top-3: {topk(ptr, ytr, 3):.1f}%")
    print(f"  Val   top-1: {topk(pva, yva, 1):.1f}%  top-3: {topk(pva, yva, 3):.1f}%")

    return {"train_top1": topk(ptr, ytr, 1), "train_top3": topk(ptr, ytr, 3),
            "val_top1": topk(pva, yva, 1), "val_top3": topk(pva, yva, 3),
            "model": model}


def main():
    d = np.load("bukva_processed.npz", allow_pickle=True)
    res = train_model_detailed(d["X"], d["y"], d["is_val"], epochs=100, hidden=128, batch=32, seed=0, log_interval=10)

    np.savez("bukva_model_detailed.npz", **res["model"].p, names=d["names"], hidden=128)
    print(f"\nМодель сохранена: bukva_model_detailed.npz")


if __name__ == "__main__":
    main()
