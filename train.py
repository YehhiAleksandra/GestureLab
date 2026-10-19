#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline-классификатор жестов этапа 1: GRU на последовательностях точек кистей
(рецепт Kaggle ISLR: точки MediaPipe -> рекуррентная сеть -> метрики top-1/top-3).
Чистый numpy, обучение на CPU за минуты.
ponytail: потолок — малые датасеты и короткие окна; апгрейд-путь — torch.nn.GRU
с тем же контрактом (X: n,t,126 -> top-1/top-3), зависимости не добавляем заранее.

Пример: python train.py dataset.npz --epochs 40 --hidden 32 --save model.npz
"""

import argparse

import numpy as np


def normalize(X):
    """Координаты относительно запястья своей кисти: инвариант к положению камеры."""
    X = X.astype(np.float64).copy()
    for hand in (0, 1):
        wrist = X[:, :, hand * 63:hand * 63 + 3]
        X[:, :, hand * 63:(hand + 1) * 63] -= np.repeat(wrist, 21, axis=2)
    return X


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class GRU:
    """Однослойный GRU с softmax-головой: BPTT по всему окну, оптимизатор Adam."""

    def __init__(self, d, h, c, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d + h)
        self.p = {
            "Wz": rng.normal(0, s, (h, d + h)), "bz": np.zeros(h),
            "Wr": rng.normal(0, s, (h, d + h)), "br": np.zeros(h),
            "Wh": rng.normal(0, s, (h, d + h)), "bh": np.zeros(h),
            "Wo": rng.normal(0, 1 / np.sqrt(h), (c, h)), "bo": np.zeros(c),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}

    def forward(self, X):
        """X: (B,T,D) -> (вероятности (B,C), кэш для backward)."""
        p = self.p
        h = np.zeros((X.shape[0], p["Wh"].shape[0]))
        steps = []
        for t in range(X.shape[1]):
            x = X[:, t, :]
            cat = np.hstack([x, h])
            z = sigmoid(cat @ p["Wz"].T + p["bz"])
            r = sigmoid(cat @ p["Wr"].T + p["br"])
            g = np.tanh(np.hstack([x, r * h]) @ p["Wh"].T + p["bh"])
            h = (1 - z) * h + z * g
            steps.append((x, h, z, r, g, cat))
        return softmax(h @ p["Wo"].T + p["bo"]), steps

    def backward(self, prob, y, steps):
        """Градиенты cross-entropy по параметрам (градиент по входу не нужен)."""
        p = self.p
        b = prob.shape[0]
        d = steps[0][0].shape[1]
        h_dim = p["Wh"].shape[0]
        onehot = np.zeros_like(prob)
        onehot[np.arange(b), y] = 1.0
        dlogits = (prob - onehot) / b
        grad = {k: np.zeros_like(v) for k, v in p.items()}
        grad["Wo"] = dlogits.T @ steps[-1][1]
        grad["bo"] = dlogits.sum(axis=0)
        dh = dlogits @ p["Wo"]
        for t in reversed(range(len(steps))):
            x, h, z, r, g, cat = steps[t]
            h_prev = steps[t - 1][1] if t else np.zeros((b, h_dim))
            dz = dh * (g - h_prev) * z * (1 - z)
            dg = dh * z * (1 - g * g)
            d_rg = dg @ p["Wh"]                 # производная по [x, r*h_prev]
            dr = d_rg[:, d:] * h_prev * r * (1 - r)
            grad["Wz"] += dz.T @ cat
            grad["bz"] += dz.sum(axis=0)
            grad["Wr"] += dr.T @ cat
            grad["br"] += dr.sum(axis=0)
            grad["Wh"] += dg.T @ np.hstack([x, r * h_prev])
            grad["bh"] += dg.sum(axis=0)
            dh = dh * (1 - z) + d_rg[:, d:] * r + dz @ p["Wz"][:, d:] + dr @ p["Wr"][:, d:]
        return grad

    def step(self, grad, lr, t):
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grad[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grad[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** t)
            vh = self.v[k] / (1 - 0.999 ** t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def topk(prob, y, k):
    if len(y) == 0:
        return 0.0
    order = np.argsort(-prob, axis=1)
    return float((order[:, :k] == y[:, None]).any(axis=1).mean() * 100)


def train_model(X, y, is_val, epochs=40, hidden=32, batch=32, lr=0.01, seed=0):
    """Обучает GRU на окнах; возвращает метрики top-1/top-3 и модель."""
    xn = normalize(X)
    model = GRU(xn.shape[2], hidden, int(y.max()) + 1, seed=seed)
    rng = np.random.default_rng(seed + 1)
    tr = ~is_val
    xtr, ytr = xn[tr], y[tr]
    step_i = 0
    for _ in range(epochs):
        order = rng.permutation(len(xtr))
        for i in range(0, len(xtr), batch):
            idx = order[i:i + batch]
            prob, steps = model.forward(xtr[idx])
            model.step(model.backward(prob, ytr[idx], steps), lr, step_i := step_i + 1)
    ptr, _ = model.forward(xn[tr])
    pva, _ = model.forward(xn[is_val])
    return {"train_top1": topk(ptr, y[tr], 1), "train_top3": topk(ptr, y[tr], 3),
            "val_top1": topk(pva, y[is_val], 1), "val_top3": topk(pva, y[is_val], 3),
            "model": model}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="npz из export_dataset.py")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="куда сохранить веса модели (.npz)")
    args = ap.parse_args()

    d = np.load(args.dataset, allow_pickle=True)
    res = train_model(d["X"], d["y"], d["is_val"], epochs=args.epochs,
                      hidden=args.hidden, seed=args.seed)
    print(f"train top-1 {res['train_top1']:5.1f}%  top-3 {res['train_top3']:5.1f}%")
    print(f"val   top-1 {res['val_top1']:5.1f}%  top-3 {res['val_top3']:5.1f}%")
    if args.save:
        np.savez(args.save, **res["model"].p, names=d["names"], hidden=args.hidden)
        print(f"модель: {args.save}")


if __name__ == "__main__":
    main()
