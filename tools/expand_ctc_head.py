"""Расширение головы CTC словами канала: 1000 классов Slovo → 1676.

Зачем: 676 слов словаря Rutube-канала (690 клипов) отсутствуют в голове модели,
поэтому на фразовой съёмке модель физически не может их назвать. Скрипт создаёт
чекпойнт-заготовку: энкодер и строки старых классов переносятся байт в байт, новые
строки нулевые — до обучения новые слова проигрывают blank и не портят прежний выход.

Дальше дообучение обычным протоколом (см. отчёт этапа 15):

  python tools/expand_ctc_head.py --out models/slovo_ctc_motion_head.pt
  python tools/finetune_ctc_signer.py --init models/slovo_ctc_motion_head.pt \\
      --dict-data datasets/youtube_dict/rutube_partial.npz \\
      --out models/slovo_ctc_motion_head_all.pt --epochs 40 --steps 120 --lr 1.5e-4

Слова с «/» и «;» считаются синонимами одного класса, как в Easy_sign: если слово
канала уже покрыто синонимом, новый класс не создаётся.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from gloss_coverage import norm, split_classes  # noqa: E402


def extend_names(names, words):
    """Имена классов + новые слова канала (нормализованные, по алфавиту, без дублей)."""
    seen = set()
    for n in names:
        seen |= split_classes(n)
    add = sorted({norm(w) for w in words if norm(w)} - seen)
    return list(names) + add


def expand_rows(t, add):
    """Строки новых классов нулевые: до обучения они проигрывают blank."""
    out = torch.zeros(t.shape[0] + add, *t.shape[1:], dtype=t.dtype)
    out[:t.shape[0]] = t
    return out


def expand_checkpoint(ck, words):
    """Чекпойнт с расширенной головой; старые строки и энкодер не меняются."""
    names = list(ck['names'])
    new_names = extend_names(names, words)
    add = len(new_names) - len(names)
    state = dict(ck['model'])
    for key in ('head.weight', 'head.bias', 'aux.weight', 'aux.bias'):
        if key in state:
            state[key] = expand_rows(state[key], add)
    out = dict(ck)
    out['model'] = state
    out['names'] = new_names
    if out.get('aux'):
        out['aux'] = int(out['aux']) + add
    out['expanded_from'] = ck.get('expanded_from', len(names))
    out['expanded_added'] = add
    return out


def selftest():
    names = ['мать', 'там/туда', 'ёлка']
    new = extend_names(names, ['Мать', 'ТУДА', 'елка', 'бобр', 'Бобр', ''])
    assert new == names + ['бобр'], new
    w = torch.ones(4, 2)
    b = torch.zeros(4)
    assert torch.equal(expand_rows(w, 2)[:4], w)
    assert float(expand_rows(w, 2)[4:].abs().sum()) == 0.0
    ck = {'names': names, 'model': {'head.weight': w, 'head.bias': b,
                                    'aux.weight': torch.ones(3, 2),
                                    'aux.bias': torch.zeros(3)},
          'aux': 3, 'feat': 496,
          'arch': {'d': 2, 'layers': 1, 'heads': 1, 'dropout': 0.0}}
    out = expand_checkpoint(ck, ['бобр'])
    assert len(out['names']) == 4 and out['expanded_added'] == 1
    assert out['model']['head.weight'].shape == (5, 2)
    assert out['model']['aux.weight'].shape == (4, 2) and out['aux'] == 4
    assert torch.equal(out['model']['head.weight'][:4], w)
    assert float(out['model']['head.weight'][4].abs().sum()) == 0.0
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default=str(ROOT / 'models/slovo_ctc_motion.pt'))
    ap.add_argument('--dict-data',
                    default=str(ROOT / 'datasets/youtube_dict/rutube_partial.npz'))
    ap.add_argument('--out', default=str(ROOT / 'models/slovo_ctc_motion_head.pt'))
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    with np.load(a.dict_data) as d:
        words = [str(w) for w in d['words']]
    out = expand_checkpoint(ck, words)
    torch.save(out, a.out)
    covered = set()
    for n in out['names']:
        covered |= split_classes(n)
    clips = sum(norm(w) in covered for w in words)
    print(f"классов {len(ck['names'])} → {len(out['names'])} "
          f"(+{out['expanded_added']}); клипов канала покрыто {clips}/{len(words)}", flush=True)
    print(f'→ {a.out}', flush=True)


if __name__ == '__main__':
    main()
