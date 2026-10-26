"""CTC beam search и языковая модель глоссов (только numpy — тестируется в CI).

Логиты CTC: 0 — blank, класс c — выход (c - 1). Языковая модель строится по цепочкам
глоссов обучающего корпуса: биграммы с интерполяцией к униграммам.
"""
import numpy as np

BLANK = 0


def smooth_log_probs(logp, width=3):
    """Сгладить покадровые log-prob скользящим средним (геометрически), сохранив нормировку."""
    logp = np.asarray(logp, np.float64)
    if width <= 1 or len(logp) < 2:
        return logp
    p = np.exp(logp - logp.max(-1, keepdims=True))
    left, right = (width - 1) // 2, width - 1 - (width - 1) // 2
    padded = np.pad(p, ((left, right), (0, 0)), mode='edge')
    cs = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(padded, axis=0)])
    out = (cs[width:] - cs[:-width]) / width
    out /= np.maximum(out.sum(-1, keepdims=True), 1e-12)
    return np.log(np.maximum(out, 1e-12))


class GlossLM:
    """Биграммная модель глоссов: P(w | prev) с интерполяцией к униграммам."""

    def __init__(self, sequences, n_classes, alpha=0.5):
        self.alpha, self.n_classes = alpha, n_classes
        self.uni = np.zeros(n_classes)
        self.start = np.zeros(n_classes)
        self.bi, self.prev = {}, {}
        for seq in sequences:
            last = None
            for w in seq:
                w = int(w)
                self.uni[w] += 1
                if last is None:
                    self.start[w] += 1
                else:
                    self.bi[(last, w)] = self.bi.get((last, w), 0) + 1
                    self.prev[last] = self.prev.get(last, 0) + 1
                last = w
        self.total = self.uni.sum()

    def logp(self, w, prev=None):
        p_uni = (self.uni[w] + 0.1) / (self.total + 0.1 * self.n_classes)
        if prev is None:
            num, den = self.start[w] + self.alpha * p_uni, self.start.sum() + self.alpha
        else:
            num = self.bi.get((prev, w), 0) + self.alpha * p_uni
            den = self.prev.get(prev, 0) + self.alpha
        return float(np.log(num / den))


def ctc_beam_search(logp, beam=8, topk=32, nbest=1, lm=None, lm_weight=0.0, word_bonus=0.0):
    """Префиксный beam search (Graves). logp [T, C+1] → nbest пар (глоссы, score).

    word_bonus поощряет каждое новое слово: без него LM штрафует длину и короткие
    малоконтрастные реплики схлопываются в пустую гипотезу.
    """
    logp = np.asarray(logp, np.float64)
    beams = {(): (0.0, -np.inf, 0.0, 0.0)}  # префикс: (p blank, p класс, LM, бонус)
    for t in range(len(logp)):
        row = logp[t]
        cand = np.argpartition(-row, min(topk, row.size - 1))[:min(topk, row.size - 1)]
        cand = cand.tolist()
        if BLANK not in cand:
            cand.append(BLANK)
        keys = list(beams)
        pb = np.array([beams[p][0] for p in keys])
        pnb = np.array([beams[p][1] for p in keys])
        lms = np.array([beams[p][2] for p in keys])
        bonus = np.array([beams[p][3] for p in keys])
        ptot = np.logaddexp(pb, pnb)
        new = {}
        for i, prefix in enumerate(keys):
            for c in cand:
                lp = row[c]
                if c == BLANK:
                    slot = new.setdefault(prefix, [-np.inf, -np.inf, lms[i], bonus[i]])
                    slot[0] = np.logaddexp(slot[0], ptot[i] + lp)
                    continue
                w = c - 1
                last = prefix[-1] if prefix else None
                new_lm = lms[i] + (lm.logp(w, last) if lm is not None else 0.0)
                if w == last:
                    slot = new.setdefault(prefix, [-np.inf, -np.inf, lms[i], bonus[i]])
                    slot[1] = np.logaddexp(slot[1], pnb[i] + lp)
                    extend = pb[i] + lp
                else:
                    extend = ptot[i] + lp
                slot = new.setdefault(prefix + (w,), [-np.inf, -np.inf, new_lm,
                                                      bonus[i] + word_bonus])
                slot[1] = np.logaddexp(slot[1], extend)
        ranked = sorted(new.items(),
                        key=lambda kv: -(np.logaddexp(kv[1][0], kv[1][1])
                                         + lm_weight * kv[1][2] + kv[1][3]))
        beams = dict(ranked[:beam])
    scored = [(list(p), float(np.logaddexp(pb, pnb) + lm_weight * lms + bonus))
              for p, (pb, pnb, lms, bonus) in beams.items()]
    scored.sort(key=lambda x: -x[1])
    return scored[:nbest]


def nbest_to_lattice(hyps, max_cands=4):
    """n-best гипотезы → варианты на каждое место для Translator.lattice.

    Варианты собираются по позициям среди гипотез с самой частой длиной; если позиции
    расходятся, на место попадает слово из лучшей гипотезы.
    """
    if not hyps:
        return []
    best_len = len(hyps[0])
    out = []
    for pos in range(best_len):
        seen, cands = set(), []
        for h in hyps:
            if len(h) != best_len:
                continue
            w = h[pos]
            if w not in seen:
                seen.add(w)
                cands.append(w)
        out.append(cands[:max_cands])
    return out


def nbest_lattice_probs(hyps, scores, max_cands=4):
    """Решётка с вероятностями: масса каждого варианта по гипотезам одной длины.

    Порядок вариантов — по маргинальной вероятности, а не по первому появлению:
    промпт с процентами обещает, что первый вариант самый вероятный.
    """
    if not hyps:
        return [], []
    best_len = len(hyps[0])
    idx = [i for i, h in enumerate(hyps) if len(h) == best_len]
    w = np.exp(np.asarray([scores[i] for i in idx], np.float64))
    w /= w.sum()
    cands, probs = [], []
    for pos in range(best_len):
        acc = {}
        for i, wi in zip(idx, w):
            acc[hyps[i][pos]] = acc.get(hyps[i][pos], 0.0) + float(wi)
        top = sorted(acc.items(), key=lambda kv: -kv[1])[:max_cands]
        cands.append([c for c, _ in top])
        probs.append([round(p, 4) for _, p in top])
    return cands, probs
