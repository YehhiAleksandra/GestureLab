"""Признаки и модель распознавания изолированных жестов по ландмаркам рук (Slovo, 1000 классов).

Общий модуль для обучения (tools/train_slovo_seq.py) и живого режима: признаки обязаны
считаться одинаково. Вход — кадры [T, 2, 21, 3] (x, y, z MediaPipe), NaN — руки нет;
руки в кадре упорядочены по x запястья.
"""
import numpy as np
import torch
import torch.nn as nn

T = 32
FEAT = 2 * 21 * 3 + 2 * 21 * 3 + 2 * 2 + 2  # локальная форма + глобальная поза + запястья + наличие


def resample(seq, t=T, start=0.0, end=1.0):
    n = len(seq)
    idx = np.linspace(start * (n - 1), end * (n - 1), t).round().astype(int)
    return seq[idx]


def features(seq):
    """[t, 2, 21, 3] → [t, FEAT]."""
    seq = seq.astype(np.float32)
    present = ~np.isnan(seq[:, :, 0, 0])
    wrist = seq[:, :, :1, :]
    size = np.linalg.norm(seq[:, :, 9, :2] - seq[:, :, 0, :2], axis=-1)[..., None, None]
    local = (seq - wrist) / np.maximum(size, 1e-3)

    valid = seq[present]
    center = valid[:, :, :2].reshape(-1, 2).mean(0) if len(valid) else np.zeros(2, np.float32)
    scale = np.nanmedian(size) if present.any() else 1.0
    glob = seq.copy()
    glob[..., :2] = (seq[..., :2] - center) / max(float(scale), 1e-3) / 4
    glob[..., 2] = seq[..., 2] * 10
    out = np.concatenate([local.reshape(len(seq), -1), glob.reshape(len(seq), -1),
                          glob[:, :, 0, :2].reshape(len(seq), -1), present.astype(np.float32)], 1)
    return np.nan_to_num(out, nan=0.0)


def augment(seq, rng):
    """Аугментация сырых точек до признаков: поворот/масштаб/сдвиг, зеркало для левшей."""
    seq = seq.copy()
    if rng.random() < 0.5:  # зеркало: левша ↔ правша, порядок рук по x меняется местами
        seq[..., 0] = 1 - seq[..., 0]
        seq = seq[:, ::-1]
    a = rng.normal(0, 0.15)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    xy = (seq[..., :2] - 0.5) @ rot.T * rng.uniform(0.85, 1.15) + 0.5 + rng.normal(0, 0.05, 2)
    seq[..., :2] = xy
    seq[..., :2] += rng.normal(0, 0.003, seq[..., :2].shape)
    return seq


class SignNet(nn.Module):
    def __init__(self, n_classes, d=256, layers=4, heads=4, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FEAT, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Conv1d(d, d, 5, padding=2, groups=4)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x):
        h = self.inp(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2) + self.pos[:, :h.shape[1]]
        return self.head(self.enc(h).mean(1))
