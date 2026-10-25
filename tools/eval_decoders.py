"""Сравнение CTC-декодеров на одних и тех же потоках: greedy, сглаживание, beam, beam+LM.

Логиты считаются один раз, дальше каждый декодер видит ровно те же вероятности — иначе
разница между декодерами смешивалась бы с разницей входов. LM строится по цепочкам
обучающей части корпуса глоссов, вес fusion подбирается на отдельном dev-наборе.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ctc_slovo import (
    batch_features,
    corpus_phrases,
    edit_counts,
    prepare_slovo,
    samples_corpus,
    samples_random,
)

from sign_beam import GlossLM, ctc_beam_search, smooth_log_probs
from sign_stream import load_ctc


def greedy(logp):
    ids, out, prev = logp.argmax(-1).tolist(), [], None
    for i in ids:
        if i != prev and i != 0:
            out.append(i - 1)
        prev = i
    return out


@torch.no_grad()
def collect(model, samples, dev):
    model.eval()
    out = []
    for stream, labels in samples:
        x, lens = batch_features([stream])
        logits = model(x.to(dev), lens.to(dev))[0].cpu()
        out.append((logits.log_softmax(-1).numpy(), labels))
    return out


def score(items, decode):
    """Метрики декодера; если decode вернул n-best, считается и oracle_exact по нему."""
    pos_ok = total = exact = oracle = dist = 0
    t0 = time.time()
    for logp, labels in items:
        out = decode(logp)
        hyps = out if out and isinstance(out[0], list) else [out]
        pred = hyps[0]
        pos_ok += sum(a == b for a, b in zip(pred, labels))
        total += len(labels)
        exact += int(pred == labels)
        oracle += int(any(h == labels for h in hyps))
        dist += edit_counts(pred, labels)
    return {'gloss_acc': pos_ok / total, 'exact_acc': exact / len(items),
            'oracle_exact': oracle / len(items), 'edit_rate': dist / total,
            'n_streams': len(items), 'n_glosses': total, 'sec': round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/slovo_ctc_corpus.pt')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--smooth', type=int, default=3)
    ap.add_argument('--lm-weights', default='0.1,0.2,0.5')
    ap.add_argument('--word-bonuses', default='0,2,4,6,8')
    ap.add_argument('--dev-seed', type=int, default=777)
    ap.add_argument('--out', default='reports/evidence/decoder_eval.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = prepare_slovo('datasets/slovo/slovo_holistic.npz')
    model, names, _ = load_ctc(a.model, dev)
    index = {n.lower(): i for i, n in enumerate(names)}
    phrases_train = corpus_phrases(index, 'train')
    phrases_test = [p for p in corpus_phrases(index, 'test') if all(c in d['te_by_class'] for c in p)]
    # Корпус почти весь дублируется: 684 из 690 тестовых фраз дословно есть в train, и LM
    # на таком совпадении просто запоминает ответ. LM учим на 80% уникальных фраз,
    # а меряем только на отложенных — их тестовые записи в LM не входили.
    uniq = sorted({tuple(p) for p in phrases_train}, key=lambda t: (len(t), t))
    perm = np.random.default_rng(2027).permutation(len(uniq))
    held = {uniq[i] for i in perm[:len(uniq) // 5]}
    lm_seqs = [list(uniq[i]) for i in perm[len(uniq) // 5:]]
    phrases_held = [p for p in phrases_test if tuple(p) in held]
    order = np.random.default_rng(2026).permutation(len(phrases_held))
    phrases_dev = [phrases_held[i] for i in order[:60]]
    phrases_test = [phrases_held[i] for i in order[60:]]
    lm = GlossLM(lm_seqs, n_classes=len(names))
    print(f'модель {a.model}: {len(names)} классов; LM по {len(lm_seqs)} уникальным фразам; '
          f'отложено {len(held)} фраз; dev {len(phrases_dev)}, тест {len(phrases_test)}',
          flush=True)

    clean = collect(model, samples_random(d['te_by_class'], d['te_classes'],
                                          np.random.default_rng(1234), a.n), dev)
    corpus = collect(model, samples_corpus(d['te_by_class'], phrases_test,
                                           np.random.default_rng(1234), a.n), dev)
    devset = collect(model, samples_corpus(d['te_by_class'], phrases_dev,
                                           np.random.default_rng(a.dev_seed), 100), dev)

    def beam_hyps(logp, w=0.0, b=0.0, nbest=1):
        return [h for h, _ in ctc_beam_search(logp, beam=a.beam, topk=a.topk, nbest=nbest,
                                              lm=lm if w else None, lm_weight=w,
                                              word_bonus=b)]

    grid = [(float(w), float(b)) for w in a.lm_weights.split(',')
            for b in a.word_bonuses.split(',')]
    tune = {f'{w:g}/{b:g}': score(devset, lambda lp, w=w, b=b: beam_hyps(lp, w, b)[0])['gloss_acc']
            for w, b in grid}
    best_w, best_b = (float(x) for x in max(tune, key=tune.get).split('/'))
    print('dev gloss_acc по (LM/бонус):', {k: round(v, 3) for k, v in tune.items()},
          f'→ {best_w:g}/{best_b:g}', flush=True)

    configs = {
        'greedy': lambda lp: greedy(lp),
        f'smooth_greedy_{a.smooth}': lambda lp: greedy(smooth_log_probs(lp, a.smooth)),
        f'beam_{a.beam}': lambda lp: beam_hyps(lp)[0],
        f'beam_{a.beam}_lm_{best_w:g}_bonus_{best_b:g}':
            lambda lp: beam_hyps(lp, best_w, best_b, nbest=5),
    }
    res = {'model': a.model, 'beam': a.beam, 'topk': a.topk, 'smooth': a.smooth,
           'lm_weight_dev': {k: round(v, 4) for k, v in tune.items()},
           'lm_weight': best_w, 'word_bonus': best_b,
           'lm_train_phrases': len(lm_seqs), 'lm_heldout_phrases': len(held),
           'lm_dev_phrases': len(phrases_dev),
           'lm_test_phrases': len(phrases_test), 'clean': {}, 'corpus': {}}
    for name, fn in configs.items():
        res['clean'][name] = score(clean, fn)
        res['corpus'][name] = score(corpus, fn)
        c, k = res['clean'][name], res['corpus'][name]
        print(f'{name:24s} clean {c["gloss_acc"]:.3f} edit {c["edit_rate"]:.3f} | '
              f'corpus {k["gloss_acc"]:.3f} edit {k["edit_rate"]:.3f} '
              f'oracle {k["oracle_exact"]:.3f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'сохранено {a.out}', flush=True)


if __name__ == '__main__':
    main()
