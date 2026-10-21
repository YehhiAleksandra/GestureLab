"""Сквозная оценка: жесты (ландмарки Slovo test) → SignNet → глоссы → LLM → русская фраза.

Фраза из тестовой части глоссового корпуса собирается из тестовых клипов Slovo, по клипу на
глоссу (по возможности от людей, которых нет в train). Границы жестов известны — это
соответствует живому режиму с паузой между жестами; коартикуляции в склейке нет.
Сравниваются три входа LLM: эталонные глоссы, top-1 распознавателя, решётка top-3.
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gloss_coverage import norm, split_classes  # noqa: E402
from gloss_to_text import Translator, chrf, load_corpus  # noqa: E402
from slovo_model import SignNet  # noqa: E402
from train_slovo_seq import load, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/slovo/slovo_seq.npz')
    ap.add_argument('--model', default='models/slovo_signnet.pt')
    ap.add_argument('-n', type=int, default=150)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='reports/evidence/end_to_end_eval.json')
    a = ap.parse_args()

    seqs, y, train, users, names = load(a.data)
    ck = torch.load(a.model)
    assert ck['names'] == names
    gloss2cls = {}
    for c, name in enumerate(names):
        for g in split_classes(name):
            gloss2cls.setdefault(g, c)

    tr_users = set(users[train])
    te = np.where(~train)[0]
    pool = {}
    for i in te:
        pool.setdefault(y[i], {True: [], False: []})[users[i] not in tr_users].append(i)

    rnd = random.Random(a.seed)
    pairs = [p for p in load_corpus('test')
             if all(norm(w) in gloss2cls and gloss2cls[norm(w)] in pool for w in p['rsl'].split())]
    print(f'фраз, где все глоссы есть в Slovo test: {len(pairs)}')
    rnd.shuffle(pairs)
    pairs = pairs[:a.n]

    clips, gold, unseen = [], [], 0
    for p in pairs:
        cls = [gloss2cls[norm(w)] for w in p['rsl'].split()]
        ids = []
        for c in cls:
            src = pool[c][True] or pool[c][False]
            unseen += bool(pool[c][True])
            ids.append(rnd.choice(src))
        clips.append(ids)
        gold.append(cls)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SignNet(len(names)).to(dev)
    model.load_state_dict(ck['model'])
    flat = [i for ids in clips for i in ids]
    probs = predict(model, [seqs[i] for i in flat], dev).softmax(1)
    topk = probs.topk(a.k, 1).indices.tolist()
    it = iter(topk)
    cands = [[next(it) for _ in ids] for ids in clips]

    n_signs = len(flat)
    top1 = sum(c[0] == g for cs, gs in zip(cands, gold) for c, g in zip(cs, gs)) / n_signs
    topk_hit = sum(g in c for cs, gs in zip(cands, gold) for c, g in zip(cs, gs)) / n_signs
    print(f'жестов {n_signs} (от незнакомых людей {unseen}), top-1 {top1:.3f}, top-{a.k} {topk_hit:.3f}')

    tr = Translator()
    t0 = time.time()

    def one(j):
        # corrected: человек клавишами 1-3 выбрал верный вариант, если он есть среди показанных
        fixed = [[names[g]] if g in c else [names[x] for x in c] for c, g in zip(cands[j], gold[j])]
        try:
            return {'oracle': tr(pairs[j]['rsl']),
                    'top1': tr(' '.join(names[c[0]] for c in cands[j])),
                    'lattice': tr.lattice([[names[x] for x in c] for c in cands[j]]),
                    'corrected': tr.lattice(fixed)}
        except Exception as e:  # один вызов без ретраев: ошибка в отчёт, не маскируется
            return {'error': str(e)}

    with ThreadPoolExecutor(8) as ex:
        hyps = list(ex.map(one, range(len(pairs))))

    rows, score = [], {'oracle': [], 'top1': [], 'lattice': [], 'corrected': []}
    for p, cs, gs, h in zip(pairs, cands, gold, hyps):
        row = {'rsl': p['rsl'], 'ref': p['russian'],
               'recognized': [[names[x] for x in c] for c in cs],
               'signs_ok': sum(c[0] == g for c, g in zip(cs, gs)), 'signs': len(gs), **h}
        if 'error' not in h:
            for m in score:
                row[f'chrf_{m}'] = round(chrf(h[m], p['russian']), 1)
                score[m].append(row[f'chrf_{m}'])
        rows.append(row)
    errors = sum('error' in r for r in rows)
    res = {'phrases': len(pairs), 'signs': n_signs, 'signs_from_unseen_people': unseen,
           'sign_top1': top1, f'sign_top{a.k}': topk_hit,
           'exact_phrases_top1': sum(r['signs_ok'] == r['signs'] for r in rows) / len(rows),
           **{f'chrf_{m}': sum(v) / max(len(v), 1) for m, v in score.items()},
           'copy_baseline': sum(chrf(p['rsl'], p['russian']) for p in pairs) / len(pairs),
           'llm_errors': errors, 'cost_usd': tr.cost, 'seconds': round(time.time() - t0)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    Path(a.out).write_text(json.dumps({**res, 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in rows[:10]:
        if 'error' in r:
            continue
        print(f"\n  жесты:   {r['rsl']}\n  увидел:  {' '.join(c[0] for c in r['recognized'])}"
              f"\n  решётка → {r['lattice']}\n  эталон:  {r['ref']}")


if __name__ == '__main__':
    sys.exit(main())
