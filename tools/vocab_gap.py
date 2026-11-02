"""Словарный разрыв независимого бенчмарка: какие слова не дают фразам дойти до оценки.

Оценка SP-10 доходит только до тех фраз, где ВСЕ слова (после канонизации) есть среди
классов головы. Этот инструмент измеряет величину разрыва и, главное, цену его закрытия:
сколько классов надо добавить и сколько из них обеспечены клипами уже скачанных источников.
Ничего не обучает и не качает — только считает по инвентарям и checkpoint.

  python tools/vocab_gap.py --benchmark D:/hfdata/sp10_eval.json --tune D:/hfdata/sp10_tune.json
  python tools/vocab_gap.py --json reports/evidence/vocab_gap_sp10.json
  python tools/vocab_gap.py --selftest
"""
import argparse
import json
import sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from finetune_ctc_phrases import (canon_map, lemma_index, load_morph,  # noqa: E402
                                  stem)
from finetune_ctc_signer import class_index  # noqa: E402
from eval_phrase_clips import words  # noqa: E402


def phrase_gap(inventory, vocab, canon, cache=None):
    """Фразы инвентаря на (дошли до оценки, чем блокированы).

    cache — папка .npz с ландмарками: без файла фраза не оценивается ни при каком
    словаре, поэтому она не попадает ни в один список и не «разблокируется» новым
    классом.
    """
    inv = json.loads(Path(inventory).read_text(encoding='utf-8'))
    usable, blocked = [], {}
    for p in inv['phrases']:
        if not p.get('duration'):
            continue
        if cache and not (Path(cache) / f"{p['id']}.npz").exists():
            continue
        cw = [canon.get(w, w) for w in words(p['phrase'])]
        miss = frozenset(w for w in cw if w not in vocab)
        if miss:
            blocked[p['id']] = miss
        else:
            usable.append(p['id'])
    return usable, blocked


def unlock_curve(usable, blocked, total):
    """Жадная кривая: классы в порядке числа разблокируемых фраз → доля оцениваемых фраз.

    Фраза становится оцениваемой, когда добавлен её самый «поздний» блокирующий класс,
    поэтому ранг фразы = максимум рангов её слов.
    """
    freq = Counter()
    for miss in blocked.values():
        freq.update(miss)
    order = [w for w, _ in freq.most_common()]
    rank = {w: i + 1 for i, w in enumerate(order)}
    need = sorted(max(rank[w] for w in miss) for miss in blocked.values())
    out = [{'classes': 0, 'phases': len(usable), 'share': round(len(usable) / max(total, 1), 4)}]
    for a in (5, 10, 20, 30, 50, 75, 100, 119, 150, 200, 250, 300, 400, len(order)):
        u = len(usable) + bisect_right(need, a)
        out.append({'classes': a, 'phases': u, 'share': round(u / max(total, 1), 4)})
    goal = next((a for a in range(0, len(order) + 1)
                 if len(usable) + bisect_right(need, a) >= 0.4 * max(total, 1)), None)
    return order, need, out, goal


def main(a):
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    vocab = set(class_index(list(ck['names'])))
    morph = load_morph() if a.lemma else None

    inv = json.loads(Path(a.benchmark).read_text(encoding='utf-8'))
    phrases = [(p['id'], p['phrase']) for p in inv['phrases'] if p.get('duration')]
    plain = [w for _, ph in phrases for w in words(ph)]
    canon_stem = canon_map(plain, vocab)
    canon_lem = canon_map(plain, vocab, morph) if morph else None

    usable, blocked = phrase_gap(a.benchmark, vocab, canon_stem, a.cache)
    n_eval = len(usable) + len(blocked)
    print(f'голова: {len(vocab)} имён классов; бенчмарк: {len(phrases)} фраз, '
          f'с клипом в кэше: {n_eval}')
    res = {'checkpoint': str(a.checkpoint), 'benchmark': str(a.benchmark),
           'head_names': len(vocab), 'phrases': len(phrases), 'phrases_with_clip': n_eval,
           'usable_stem': len(usable), 'share_stem': round(len(usable) / max(n_eval, 1), 4)}

    if canon_lem is not None:
        u2, b2 = phrase_gap(a.benchmark, vocab, canon_lem, a.cache)
        extra = sorted({w for w in plain if canon_lem.get(w, w) != canon_stem.get(w, w)})
        print(f'доходит до оценки (канон по основе): {len(usable)}/{n_eval} '
              f'= {100 * len(usable) / n_eval:.1f}%')
        print(f'доходит до оценки (канон по лемме): {len(u2)}/{n_eval} '
              f'= {100 * len(u2) / n_eval:.1f}%  (+{len(u2) - len(usable)} фраз, '
              f'{len(extra)} форм разблокировано леммой)')
        res.update(usable_lemma=len(u2), share_lemma=round(len(u2) / n_eval, 4),
                   lemma_unlocked_forms=len(extra))
        usable, blocked, canon = u2, b2, canon_lem
    else:
        print(f'доходит до оценки (канон по основе): {len(usable)}/{n_eval} '
              f'= {100 * len(usable) / n_eval:.1f}%')
        canon = canon_stem

    order, need, curve, goal = unlock_curve(usable, blocked, n_eval)
    print(f'блокирующих типов слов: {len(order)}; классов до 40% покрытия: {goal}')
    for row in curve:
        print(f"  +{row['classes']:>4} классов → {row['phases']}/{n_eval} "
              f"= {100 * row['share']:.1f}%")

    have = set()
    if a.tune:
        tinv = json.loads(Path(a.tune).read_text(encoding='utf-8'))
        ttypes = {w for p in tinv['phrases'] if p.get('duration') for w in words(p['phrase'])}
        ttypes |= {canon.get(w, w) for w in ttypes}
        t_stems = {stem(t) for t in ttypes}
        have = {w for w in order if w in ttypes or stem(w) in t_stems}
        closable = [miss for miss in blocked.values() if miss <= set(have)]
        u_src = len(usable) + len(closable)
        print(f'  из {len(order)} типов клипы в train-сплите бенчмарка есть под {len(have)}; '
              f'фраз закрывается без новых данных: максимум {u_src}/{n_eval} '
              f'= {100 * u_src / n_eval:.1f}%')
        res.update(with_clip_types=len(have), ceiling_same_corpus=u_src,
                   ceiling_share=round(u_src / max(n_eval, 1), 4))

    freq = Counter()
    for miss in blocked.values():
        freq.update(miss)
    res['top_missing'] = [{'word': w, 'phases': freq[w]} for w in order[:60]]
    res['curve'] = curve
    res['classes_for_40pct'] = goal
    if a.list:
        # тот же порядок, что в кривой: первые N позиций = минимальный набор классов
        Path(a.list).write_text(json.dumps(
            {'benchmark': str(a.benchmark), 'canon': 'lemma' if canon_lem else 'stem',
             'classes': [{'word': w, 'phases': freq[w],
                          'clip_in_tune': (w in have) if a.tune else None}
                         for w in order]}, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'список классов для расширения ({len(order)}): {a.list}')
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'записано: {a.json}')
    return 0


class _StubMorph:
    """Минимальная подделка pymorphy3: parse(w)[0].normal_form по словарю форм."""

    def __init__(self, lemmas):
        self.lemmas = lemmas

    def parse(self, w):
        class _P:
            normal_form = self.lemmas.get(w, w)
        return [_P()]


def selftest():
    names = ['дом', 'автомобиль', 'там/туда', 'код', 'кода']
    vocab = set(class_index(names))
    plain = ['дом', 'дома', 'автомобиля', 'туда', 'клуба', 'клубу']
    canon = canon_map(plain, vocab)
    assert canon.get('дома') is None, 'короткую основу грубый стемминг не трогает'
    assert canon.get('автомобиля') == 'автомобиль' and canon.get('клуба') is None, canon
    lem = {'дома': 'дом', 'клуба': 'клуб'}
    canon2 = canon_map(plain, vocab, _StubMorph(lem))
    assert canon2.get('дома') == 'дом', 'лемма добирает то, что не взял стемминг'
    assert canon2.get('клуба') is None, 'леммы нет в голове — не мапим'
    assert canon2.get('автомобиля') == 'автомобиль'
    assert canon2.get('клубу') is None
    # одна лемма относится к двум именам головы — неоднозначно, не мапим
    assert lemma_index({'дом', 'дома'}, _StubMorph(lem)) == {}
    assert lemma_index({'дом'}, _StubMorph(lem)) == {'дом': 'дом'}

    inv = {'phrases': [{'id': 'a', 'phrase': 'Дом дома', 'duration': 1.0},
                       {'id': 'b', 'phrase': 'Клуба клуба дома', 'duration': 1.0},
                       {'id': 'c', 'phrase': 'Туда', 'duration': 1.0},
                       {'id': 'd', 'phrase': 'Клуба дома', 'duration': 1.0},
                       {'id': 'e', 'phrase': 'Дом', 'duration': None}]}
    p = Path(__file__).with_suffix('.selftest.json')
    p.write_text(json.dumps(inv), encoding='utf-8')
    try:
        usable, blocked = phrase_gap(p, vocab, canon)
        assert usable == ['c'] and set(blocked) == {'a', 'b', 'd'}, (usable, blocked)
        order, need, curve, goal = unlock_curve(usable, blocked, 4)
        assert order == ['дома', 'клуба'], order
        assert curve[1]['phases'] == 4 and goal == 1, (curve, goal)
        # без файла кэша фраза не оценивается и не считается разблокируемой
        u2, b2 = phrase_gap(p, vocab, canon, Path('__no_such_dir__'))
        assert u2 == [] and b2 == {}, (u2, b2)
    finally:
        p.unlink()
    print('selftest ok')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='D:/hfdata/sp10_eval.json')
    ap.add_argument('--tune', default='D:/hfdata/sp10_tune.json')
    ap.add_argument('--cache', default='D:/hfdata/sp10_rsl/landmarks')
    ap.add_argument('--checkpoint', default='D:/hfdata/dict_folds_honest/fold0.pt')
    ap.add_argument('--lemma', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--json', default=None)
    ap.add_argument('--list', default=None,
                    help='записать упорядоченный список отсутствующих классов для расширения головы')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    sys.exit(selftest() if args.selftest else main(args))
