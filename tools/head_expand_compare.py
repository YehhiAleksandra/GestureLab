"""Сравнение двух прогонов одной метрики по клипам: помогло ли расширение головы.

`meaning_compare` сравнивает конфиги декодера внутри одного набора фолдов. Здесь другой
вопрос: та же выборка, но другая голова (например 3413 классов против 3993 после
расширения словаря). Покрытые фразы при этом меняются — новая голова знает больше слов,
и в оценку проходят клипы, которых у старой не было вовсе. Поэтому два числа:

  1. покрытие — сколько фраз бенчмарка стало измеримым (это выигрыш сам по себе);
  2. качество на общих клипах — парный бутстрап, чтобы не сравнивать разные выборки.

  python tools/head_expand_compare.py --old reports/evidence/meaning_sp10.json \\
      --new D:/hfdata/meaning_sp10_expanded.json --config 1
  python tools/head_expand_compare.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from meaning_compare import boot_diff  # noqa: E402


def pick(doc, config):
    """Ряды одного конфига прогона → {id: ряд}; порядок фолдов не важен, пары по id."""
    rows = doc['rows'][str(config)] if str(config) in doc.get('rows', {}) else None
    if rows is None:
        sys.exit(f'в {doc.get("tool", "?")} нет конфигов {config}; есть '
                 f'{sorted(doc.get("rows", {}))}')
    by_id = {}
    for r in rows:
        if r['id'] in by_id:
            sys.exit(f'клип {r["id"]} встречается дважды — прогоны не сопоставимы')
        by_id[r['id']] = r
    return by_id


def compare(old_rows, new_rows, metric):
    """Парный бутстрап по общим клипам + что дали клипы, дошедшие только до новой головы."""
    common = sorted(set(old_rows) & set(new_rows))
    only_new = sorted(set(new_rows) - set(old_rows))
    o = [old_rows[i] for i in common]
    n = [new_rows[i] for i in common]
    get = lambda r, k: (r.get(k) or 0)  # noqa: E731 — пустой перевод судится нулём
    out = {
        'clips_old': len(old_rows), 'clips_new': len(new_rows), 'common': len(common),
        'unlocked': len(only_new),
        'unlock_word_hits': sum(get(new_rows[i], 'hits_canon') for i in only_new),
        'unlock_word_tokens': sum(new_rows[i]['total'] for i in only_new),
        'words_common': sum(r['total'] for r in o),
        'hits_old': sum(get(r, 'hits_canon') for r in o),
        'hits_new': sum(get(r, 'hits_canon') for r in n),
        'vs_old': {k: boot_diff([get(r, k) for r in n], [get(r, k) for r in o])
                   for k in metric}}
    for k in metric:
        d = out['vs_old'][k]
        d['wins'] = sum(1 for a, b in zip(n, o) if get(a, k) > get(b, k))
        d['losses'] = sum(1 for a, b in zip(n, o) if get(a, k) < get(b, k))
    return out


METRICS = {'word_hits_canon': 'hits_canon', 'meaning': 'meaning', 'chrf': 'chrf'}


def run(a):
    old = json.loads(Path(a.old).read_text(encoding='utf-8'))
    new = json.loads(Path(a.new).read_text(encoding='utf-8'))
    res = compare(pick(old, a.old_config), pick(new, a.config),
                  [METRICS[m] for m in a.metrics.split(',')])
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {'tool': 'head_expand_compare', 'old': str(a.old), 'new': str(a.new),
             'config': a.config, **res}, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'→ {a.out}')
    return res


def selftest():
    key = lambda p, h, m: {'id': p, 'hits_canon': h, 'total': 2,  # noqa: E731
                           'meaning': m, 'chrf': m * 10.0}
    old = {str(i): key(i, 0, 0) for i in range(4)}
    new = {**{str(i): key(i, 1, 1) for i in range(4)}, '4': key(4, 2, 2), '5': key(5, 0, 0)}
    r = compare(old, new, ['hits_canon', 'meaning'])
    assert r['common'] == 4 and r['unlocked'] == 2, r
    assert r['hits_old'] == 0 and r['hits_new'] == 4, r
    assert r['unlock_word_hits'] == 2 and r['unlock_word_tokens'] == 4, r
    assert r['vs_old']['meaning']['wins'] == 4 and r['vs_old']['meaning']['losses'] == 0, r
    assert r['vs_old']['meaning']['distinguishable'], '4 из 4 лучше — должно различиться'
    same = compare(old, dict(old), ['meaning'])
    assert not same['vs_old']['meaning']['distinguishable'], same['vs_old']['meaning']
    print('selftest ok')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', default='', help='прогон до расширения (meaning_compare)')
    ap.add_argument('--new', default='', help='прогон после')
    ap.add_argument('--config', type=int, default=1, help='индекс конфига в обоих прогонах')
    ap.add_argument('--old-config', type=int, default=None)
    ap.add_argument('--metrics', default='meaning,chrf,word_hits_canon')
    ap.add_argument('--out', default='')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    a.old_config = a.config if a.old_config is None else a.old_config
    if not (a.old and a.new):
        ap.error('--old и --new обязательны (или --selftest)')
    run(a)


if __name__ == '__main__':
    main()
