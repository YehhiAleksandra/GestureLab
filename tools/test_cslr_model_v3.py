"""Проверка CSLR v3: словарь и blank берутся из чекпойнта, а не пересобираются."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_cslr_model_v3 import CSLRModelV3


def ctc_greedy(logits, blank):
    ids = logits.argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def main():
    ckpt = torch.load('models/cslr_model_v3.pt', map_location='cpu')
    vocab, inv, blank = ckpt['vocab'], ckpt['inv_vocab'], ckpt['blank_idx']
    inv = {int(k): v for k, v in inv.items()}

    model = CSLRModelV3(input_dim=126, hidden_dim=512, num_layers=3, num_classes=len(vocab))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with open('datasets/youtube_rsl/cslr_dataset_with_real_landmarks.json', encoding='utf-8') as f:
        segments = json.load(f)['segments']

    hit = empty = 0
    preds = Counter()
    with torch.no_grad():
        for seg in segments:
            x = torch.tensor(np.array(seg['features'], dtype=np.float32)).unsqueeze(0)
            dec = ctc_greedy(model(x, torch.tensor([x.shape[1]]))[0], blank)
            words = [inv[i] for i in dec]
            preds.update(words)
            empty += not words
            hit += seg['gesture'] in words

    n = len(segments)
    print(f'Сегментов: {n} (это обучающая выборка, отложенной нет)')
    print(f'Попадание метки в декод: {hit}/{n} = {100 * hit / n:.1f}%')
    print(f'Пустых декодов: {empty}/{n}')
    print(f'Разных предсказанных слов: {len(preds)}; топ-10: {preds.most_common(10)}')
    print(f'Частота самой частой метки (бейзлайн): '
          f'{100 * Counter(s["gesture"] for s in segments).most_common(1)[0][1] / n:.1f}%')


if __name__ == '__main__':
    main()
