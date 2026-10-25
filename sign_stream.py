"""CTC-модель непрерывного РЖЯ и декодирование (нужен torch).

Признаки и нарезка реплик — в sign_utils (только numpy, проверяются в CI).
Вход модели — holistic-кадры [T, NPTS, 3], как их собирает live_signs.holistic_frame.
"""
import numpy as np
import torch
from torch import nn

from sign_utils import FEAT

BLANK = 0  # в выходах CTC blank всегда первый, классы сдвинуты на +1
MAX_STEPS = 1024  # позиционное кодирование: 1024 шага ≈ 80 с после прореживания ×2


class CTCSignNet(nn.Module):
    """Покадровый энкодер → CTC-логиты. Прореживает время вдвое (25 → 12.5 шагов/с)."""

    def __init__(self, n_classes, d=256, layers=4, heads=4, dropout=0.15, feat=FEAT,
                 max_steps=MAX_STEPS):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(feat, d), nn.LayerNorm(d), nn.GELU(),
                                 nn.Dropout(dropout))
        self.conv = nn.Conv1d(d, d, 5, padding=2, groups=4)
        self.down = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, max_steps, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, n_classes + 1)

    def forward(self, x, lengths=None):
        h = self.inp(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.down(h.transpose(1, 2)).transpose(1, 2)
        h = h + self.pos[:, :h.shape[1]]
        mask = None
        if lengths is not None:
            steps = (lengths + 1) // 2
            mask = torch.arange(h.shape[1], device=x.device)[None] >= steps[:, None]
        return self.head(self.enc(h, src_key_padding_mask=mask))


def out_lengths(lengths):
    return (np.asarray(lengths) + 1) // 2


def greedy_decode(logits):
    """Логиты [T, C+1] → (индексы классов без blank, уверенность каждого)."""
    p = logits.softmax(-1)
    ids = p.argmax(-1).tolist()
    conf = p.max(-1).values.tolist()
    out, scores = [], []
    prev = None
    for i, c in zip(ids, conf):
        if i != prev and i != BLANK:
            out.append(i - 1)
            scores.append(c)
        prev = i
    return out, scores


def load_ctc(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    model = CTCSignNet(len(ck['names']), feat=ck.get('feat', FEAT), **ck.get('arch', {}))
    model.load_state_dict(ck['model'])
    return model.to(device).eval(), ck['names'], ck
