"""LLM-перевод с фьюжн-контекстом: цепочки n-best + центроидные кандидаты (этап 21, п.2).

Реранк-мешок (этап 21) поднял мешочный recall 23 → 27%, но production-метрика —
«смысл» через LLM. Здесь сравниваются два контекста одной модели на тех же клипах:
  A — как в production: топ-5 цепочек n-best (этапы 19–20, «смысл ≥1» 17.9%);
  B — то же + псевдо-цепочка из центроидных кандидатов (топ-5 близости к словарю).
Судья 0/1/2 и chrF считаются для обоих вариантов, стоимость прогона — в отчёте.
Пустой top-1 не идёт в LLM (B50): meaning=0 у обоих вариантов.

Вход — reports/evidence/rerank_eval.json (chains и aff_top уже сохранены по клипам).

  python tools/fusion_translate.py --selftest
  python tools/fusion_translate.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dialogue_state import translate_in_context  # noqa: E402
from gloss_to_text import SYSTEM, Translator, chrf  # noqa: E402


def fusion_chains(chains, aff_top):
    """Контекст B: цепочки + центроидные кандидаты последней псевдо-цепочкой."""
    cand = [w for w, _ in aff_top]
    return list(chains) + [cand] if cand else list(chains)


def summarize(rows, key):
    judged = [r for r in rows if r.get(key + '_meaning') is not None]
    n = max(len(rows), 1)
    return {'judged': len(judged),
            'meaning_ge1': round(sum(r[key + '_meaning'] >= 1 for r in judged)
                                 / max(len(judged), 1), 3),
            'meaning_2': round(sum(r[key + '_meaning'] == 2 for r in judged)
                               / max(len(judged), 1), 3),
            'chrf': round(float(sum(r[key + '_chrf'] for r in rows)) / n, 2),
            'exact': sum(1 for r in rows if r.get(key + '_exact'))}


def run(a):
    data = json.loads(Path(a.src).read_text(encoding='utf-8'))
    rows = data['rows']
    if a.limit:
        rows = rows[:a.limit]
    tr = Translator()
    t0 = time.time()
    for i, r in enumerate(rows):
        chains = r.get('chains') or []
        top1 = chains[0] if chains else []
        variants = {'a': chains, 'b': fusion_chains(chains, r.get('aff_top') or [])}
        if not top1:  # B50: пустой top-1 — в LLM не идёт, смысл 0 у обоих
            for v in variants:
                r[v + '_translation'], r[v + '_meaning'] = '', 0
                r[v + '_chrf'], r[v + '_exact'] = 0.0, False
            continue
        glosses = [(w, 0.5) for w in top1]
        for v, ch in variants.items():
            text = translate_in_context(tr, glosses, [], SYSTEM, chains=ch)
            r[v + '_translation'] = text
            r[v + '_meaning'] = tr.judge(text, r['phrase']) if text else 0
            r[v + '_chrf'] = round(chrf(text, r['phrase']), 3)
            r[v + '_exact'] = text.strip('. ').lower() == r['phrase'].strip('. ').lower()
        if (i + 1) % 10 == 0:  # прогресс не теряется при сбое в конце прогона
            Path(a.out).write_text(json.dumps(
                {'tool': 'fusion_translate', 'date': time.strftime('%Y-%m-%d'),
                 'src': str(a.src), 'summary': {'partial': i + 1}, 'rows': rows},
                ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"{i + 1}/{len(rows)} {r['id'][:8]} {r['phrase']!r}: "
              f"A={r['a_translation']!r} ({r['a_meaning']}) | "
              f"B={r['b_translation']!r} ({r['b_meaning']})", flush=True)

    summary = {'clips': len(rows), 'a': summarize(rows, 'a'), 'b': summarize(rows, 'b'),
               'cost_usd': round(tr.cost, 4),
               'seconds': round(time.time() - t0, 1)}
    out = {'tool': 'fusion_translate', 'date': time.strftime('%Y-%m-%d'),
           'src': str(a.src), 'summary': summary, 'rows': rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f'→ {a.out}', flush=True)


def selftest():
    ch = [['мать', 'я'], ['нога']]
    # кандидаты добавляются одной псевдо-цепочкой в конец, пустые не добавляются
    fused = fusion_chains(ch, [('вода', 0.9), ('рука', 0.8)])
    assert fused == [['мать', 'я'], ['нога'], ['вода', 'рука']]
    assert fusion_chains(ch, []) == ch
    rows = [{'a_meaning': 1, 'b_meaning': 2, 'a_chrf': 10.0, 'b_chrf': 30.0,
             'a_exact': False, 'b_exact': True}]
    s = summarize(rows, 'b')
    assert s['meaning_ge1'] == 1.0 and s['meaning_2'] == 1.0 and s['chrf'] == 30.0
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='reports/evidence/rerank_eval.json')
    ap.add_argument('--out', default='reports/evidence/fusion_translate.json')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)


if __name__ == '__main__':
    main()