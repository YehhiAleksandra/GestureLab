"""Дообучение CTC на фразовых клипах канала — единственный размеченный видео-ресурс.

Камеры нет: учимся только на 67 клипах Rutube (259 с, один исполнитель, точная фраза
в заголовке). Голова расширяется словами этих фраз, CTC учится выдавать цепочку
текстовых слов, Slovo подмешивается, чтобы не разучиться остальному.

Честная проверка — 5 фолдов: каждый клип предсказывает модель, которая его не видела.
Сравнение с базой идёт на тех же клипах. Метки — русские слова из заголовка, а не
глоссы РЖЯ: порядок и служебные слова могут не совпадать с жестами, это ограничение
протокола, а не модели.

  python tools/finetune_ctc_phrases.py --epochs 20 --steps 60
  python tools/finetune_ctc_phrases.py --no-llm --folds 2 --limit 20
  python tools/finetune_ctc_phrases.py --selftest
"""
import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dialogue_state import translate_in_context  # noqa: E402
from eval_phrase_clips import hits, words  # noqa: E402
from expand_ctc_head import expand_checkpoint  # noqa: E402
from finetune_ctc_signer import class_index  # noqa: E402
from gloss_to_text import SYSTEM, Translator, chrf  # noqa: E402
from sign_stream import CTCSignNet, greedy_decode  # noqa: E402
from sign_utils import FEAT, split_utterances  # noqa: E402
from slovo_model import augment, resample  # noqa: E402
from train_ctc_slovo import (batch_features, corpus_phrases, evaluate, file_sha256,  # noqa: E402
                             git_commit, make_corpus_stream, make_stream, prepare_slovo,
                             spec_augment, warm_start)


STEM_END = ('иями', 'ями', 'ами', 'ов', 'ев', 'ей', 'ой', 'ый', 'ий', 'ая', 'яя', 'ое',
            'ее', 'ые', 'ие', 'ых', 'их', 'ым', 'им', 'ом', 'ем', 'ам', 'ям', 'ах', 'ях',
            'у', 'ю', 'а', 'я', 'ы', 'и', 'е', 'о', 'ь', 'й')


def stem(w):
    """Грубая основа слова: срезать падежное/родовое окончание, корень не короче 4."""
    for e in STEM_END:
        if w.endswith(e) and len(w) - len(e) >= 4:
            return w[:-len(e)]
    return w


def canon_map(plain, vocab):
    """Слово фразы → уже обученный класс по общей основе (знак не склоняется).

    «автомобиля» и «автомобиль» — один жест, но в голове это разные классы: без
    нормализации склоняемая форма учится с нуля по одному клипу, хотя у леммы есть
    словарный клип. Неоднозначную основу не трогаем — лучше ноль, чем чужая метка.
    """
    stems = {}
    for v in vocab:
        stems.setdefault(stem(v), set()).add(v)
    out = {}
    for w in plain:
        if w in vocab:
            out[w] = w
            continue
        cand = stems.get(stem(w), set()) - {w}
        if len(cand) == 1:
            out[w] = next(iter(cand))
    return out


def load_phrase_clips(cache, inventory, names, canon=None):
    """Кэш ландмарков → [(клип)]: самый длинный отрезок с руками, метки = слова фразы.

    canon — карта канонизации; метки и оценка идут по каноническим словам, исходная
    фраза остаётся для перевода и судьи смысла.
    """
    inv = json.loads(Path(inventory).read_text(encoding='utf-8'))
    index = class_index(names)
    clips, skipped = [], []
    for p in inv['phrases']:
        if not p.get('duration'):
            continue
        path = Path(cache) / f"{p['id']}.npz"
        if not path.exists():
            skipped.append({'id': p['id'], 'reason': 'нет кэша'})
            continue
        with np.load(path) as d:
            frames, fps = d['frames'], float(d['fps'])
        spans = split_utterances(frames, fps, pause=0.7, min_len=0.4, max_len=15.0)
        a, b = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(frames))
        ws = words(p['phrase'])
        cw = [canon.get(w, w) for w in ws] if canon else ws
        ids = [index.get(w) for w in cw]
        if not ws or any(i is None for i in ids):
            skipped.append({'id': p['id'], 'reason': 'слово вне головы'})
            continue
        span = frames[a:b].astype(np.float32)
        min_out = len(ids) + sum(x == y for x, y in zip(ids, ids[1:]))
        if (len(span) + 1) // 2 < min_out:
            skipped.append({'id': p['id'], 'reason': f'кадров {len(span)} < меток {len(ids)}'})
            continue
        clips.append({'id': p['id'], 'phrase': p['phrase'], 'words': ws,
                      'canon_words': cw, 'canon': ' '.join(cw),
                      'span': span, 'labels': ids, 'frames': int(len(frames)),
                      'used': [int(a), int(b)]})
    return clips, skipped


def folds(n, k, seed):
    """Детерминированное разбиение индексов: каждый клип ровно в одном фолде."""
    order = np.random.default_rng(seed).permutation(n)
    return [sorted(order[i::k].tolist()) for i in range(k)]


def training_clips(clips, held_idx, allowed_ids=None):
    """Отфильтровать только train; held всегда остаётся полным для честного CV."""
    held = set(held_idx)
    return [c for i, c in enumerate(clips)
            if i not in held and (allowed_ids is None or c['id'] in allowed_ids)]


def production_counts(clips, prod_clips):
    """Раздельные счётчики загруженного набора и реально обучавшихся клипов."""
    return {'clips': len(clips), 'train_clips': len(prod_clips), 'held_clips': 0}


def matched(pred, phrase):
    """Совпавшие слова как мешок: Counter пересечения предсказания и эталона."""
    p = Counter(words(' '.join(pred)))
    r = Counter(words(phrase))
    return p & r


def augment_hard(seq, rng):
    """Жёсткая аугментация для одного примера на слово: темп, пропуски кадров, части.

    Один клип на слово не даёт вариативности контекста, поэтому инвариантность
    приходится выжимать из искажений: смена темпа 0,8–1,25, выпадение 8% кадров
    (как пропуски детекции) и выпадение лица/позы/одной руки.
    """
    seq = augment(seq, rng)
    n = len(seq)
    if n > 8:
        seq = resample(seq, max(6, int(round(n * rng.uniform(0.8, 1.25)))))
    if len(seq) > 10 and rng.random() < 0.3:
        keep = np.ones(len(seq), bool)
        drop = rng.choice(np.arange(1, len(seq)), size=max(1, int(0.08 * len(seq))),
                          replace=False)
        keep[drop] = False
        seq = seq[keep]
    if rng.random() < 0.3:
        part = int(rng.integers(4))
        a, b = [(0, 21), (21, 42), (42, 49), (49, 59)][part]
        seq[:, a:b] = np.nan
    return seq


def ctc_align(logp, labels):
    """Viterbi-выравнивание CTC: [S, C+1] логитов и метки → [(метка, s, e)] по шагам.

    Шаги логитов идут вдвое реже кадров (down. stride=2), поэтому вызывающий код
    умножает границы на 2.
    """
    logp = np.asarray(logp, np.float64)
    S = len(logp)
    ext = [0]
    for c in labels:
        ext += [int(c) + 1, 0]
    N = len(ext)
    dp = np.full((S, N), -1e9)
    ptr = np.zeros((S, N), np.int8)
    dp[0, 0] = logp[0, 0]
    if N > 1:
        dp[0, 1] = logp[0, ext[1]]
    for t in range(1, S):
        for n in range(N):
            best, bp = dp[t - 1, n], 0
            if n >= 1 and dp[t - 1, n - 1] > best:
                best, bp = dp[t - 1, n - 1], 1
            if n >= 2 and ext[n] != 0 and ext[n] != ext[n - 2] and dp[t - 1, n - 2] > best:
                best, bp = dp[t - 1, n - 2], 2
            dp[t, n] = best + logp[t, ext[n]]
            ptr[t, n] = bp
    n = N - 1 if dp[S - 1, N - 1] >= dp[S - 1, max(N - 2, 0)] else N - 2
    path = np.zeros(S, np.int32)
    for t in range(S - 1, -1, -1):
        path[t] = n
        n -= int(ptr[t, n])
    spans = []
    for i, c in enumerate(labels):
        ts = np.where(path == 2 * i + 1)[0]
        if len(ts):
            spans.append((int(c), int(ts[0]), int(ts[-1]) + 1))
    return spans


@torch.no_grad()
def align_clips(path, clips, dev='cpu'):
    """Клипы → отрезки слов по Viterbi-выравниванию модели (кадры, не шаги логитов)."""
    from sign_stream import load_ctc
    model, names, ck = load_ctc(str(path), dev)
    feat = ck.get('feat', FEAT)
    out = []
    for c in clips:
        x, lens = batch_features([c['span']], feat)
        logp = model(x.to(dev), lens.to(dev))[0].cpu().log_softmax(-1).numpy()
        segs = []
        for lab, a, b in ctc_align(logp, c['labels']):
            a2, b2 = 2 * a, min(2 * b, len(c['span']))
            if b2 - a2 >= 2:
                segs.append((lab, c['span'][a2:b2], c['id']))
        out.append(segs)
    return out


def build_pool(aligned):
    """Выровненные отрезки → {класс: [(кадры, id клипа)]} для пересборки фраз."""
    pool = {}
    for segs in aligned:
        for lab, seg, cid in segs:
            pool.setdefault(lab, []).append((seg, cid))
    return pool


def recombine(pool, rng, wmin=2, wmax=4):
    """Синтетическая фраза: отрезки разных слов подряд, между ними 2–4 кадра паузы."""
    words = list(pool)
    k = int(rng.integers(wmin, wmax + 1))
    labels = [int(w) for w in rng.choice(words, size=k)]
    parts = []
    for i, lab in enumerate(labels):
        seg = pool[lab][int(rng.integers(len(pool[lab])))][0]
        if len(seg) > 6:  # срезать переходные кадры на краях
            seg = seg[1:-1]
        parts.append(seg)
        if i < len(labels) - 1:
            parts.append(np.repeat(seg[-1:], int(rng.integers(2, 5)), axis=0))
    return np.concatenate(parts).astype(np.float32), labels


def mix_phrase(seqs_by_class, classes, corpus, phrases, rng, corpus_frac=0.30,
               phrase_frac=0.45, single_frac=0.10, aug=augment, pool=None,
               recombine_frac=0.0):
    """CTC-пример: пересобранная фраза, цепочка корпуса, клип, одиночный Slovo, поток."""
    r = rng.random()
    if pool and r < recombine_frac:
        s, labels = recombine(pool, rng)
        return aug(s, rng), labels
    if recombine_frac:
        r = (r - recombine_frac) / (1 - recombine_frac)
    if corpus and r < corpus_frac:
        return make_corpus_stream(seqs_by_class, corpus, rng)
    if phrases and r < corpus_frac + phrase_frac:
        s, labels = phrases[int(rng.integers(len(phrases)))]
        return aug(s, rng), list(labels)
    if r < corpus_frac + phrase_frac + single_frac:
        c = int(rng.choice(classes))
        pool = seqs_by_class[c]
        s = augment(pool[int(rng.integers(len(pool)))], rng)
        return resample(s, int(rng.integers(16, 49))), [c]
    return make_stream(seqs_by_class, classes, rng)


@torch.no_grad()
def greedy_rows(model, clips, names, vocab, dev='cpu'):
    """Greedy по клипу: глоссы, попадания, пустые; vocab — слова обучения этого фолда."""
    model.eval()
    rows = []
    for c in clips:
        x, lens = batch_features([c['span']], model.inp[0].in_features)
        pred, conf = greedy_decode(model(x.to(dev), lens.to(dev))[0].cpu())
        glosses = [names[i] for i in pred]
        m = matched(glosses, c['canon'])
        rows.append({'id': c['id'], 'phrase': c['phrase'], 'canon': c['canon'],
                     'pred': glosses, 'conf': [round(v, 3) for v in conf],
                     'glosses': len(glosses), 'hits': sum(m.values()),
                     'hits_strict': hits(glosses, c['phrase'])[0],
                     'total': len(c['words']),
                     'seen_hits': sum(v for w, v in m.items() if w in vocab),
                     'seen_total': sum(w in vocab for w in c['canon_words'])})
    return rows


def summarize(rows):
    n = max(len(rows), 1)
    hits_n = sum(r['hits'] for r in rows)
    tokens = sum(r['total'] for r in rows)
    strict_hits = sum(r.get('hits_strict', r['hits']) for r in rows)
    seen_hits = sum(r['seen_hits'] for r in rows)
    seen_tokens = sum(r['seen_total'] for r in rows)
    return {'clips': len(rows),
            'empty': round(sum(r['glosses'] == 0 for r in rows) / n, 3),
            'glosses_mean': round(float(np.mean([r['glosses'] for r in rows])), 2),
            'word_hits': hits_n, 'word_tokens': tokens,
            'word_recall': round(hits_n / max(tokens, 1), 3),
            'word_hits_strict': strict_hits,
            'word_recall_strict': round(strict_hits / max(tokens, 1), 3),
            'seen_hits': seen_hits, 'seen_tokens': seen_tokens,
            'seen_recall': round(seen_hits / max(seen_tokens, 1), 3),
            # все слова эталона найдены (мешок, без учёта порядка и лишних слов)
            'phrase_all_words': sum(r['hits'] == r['total'] and r['total'] > 0 for r in rows)}


def train_fold(names, init, arch, feat, train_clips, seqs_by_class, classes, corpus, a,
               fold, dev):
    """Один фолд: тёплый старт от расширенной головы, CTC по фразам и Slovo."""
    torch.manual_seed(a.seed + fold)
    rng = np.random.default_rng(a.seed + fold)
    model = CTCSignNet(len(names), d=arch['d'], layers=arch['layers'],
                       dropout=arch['dropout'], feat=feat, aux=0).to(dev)
    warm_start(model, init, names)
    if a.freeze_encoder:
        for name, p in model.named_parameters():
            if not name.startswith('head.'):
                p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.steps,
                                                pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    pool = [(c['span'], c['labels']) for c in train_clips]
    if a.dup_times > 0 and a.dup_ids:
        dup = [(c['span'], c['labels']) for c in train_clips if c['id'] in a.dup_ids]
        pool += dup * a.dup_times
        print(f'  оверсэмплинг: {len(dup)} клипов ×{a.dup_times} '
              f'(доля в пуле {len(dup) * a.dup_times}/{len(pool)})', flush=True)
    corpus_frac = a.corpus_frac if corpus else 0.0
    phrase_frac = a.phrase_frac + (a.corpus_frac if not corpus else 0.0)
    seg_pool = None
    if a.recombine_frac > 0:
        apath = Path(a.align_from.format(fold=fold))
        if not apath.exists():
            sys.exit(f'нет модели для выравнивания {apath}')
        seed = torch.load(apath, map_location='cpu', weights_only=False).get('config', {}).get('seed')
        if seed != a.seed:
            sys.exit(f'выравнивание из {apath} обучено с seed {seed}, а фолды с {a.seed}: '
                     f'отложенные клипы могли попасть в обучение')
        # голова выравнивания должна покрывать все классы новых меток: тёплый старт
        # старого фолда на расширенную голову (иначе ctc_align падает с IndexError)
        align_exp = Path(a.work) / f'align{fold}_expanded.pt'
        if not align_exp.exists():
            m = CTCSignNet(len(names), d=arch['d'], layers=arch['layers'],
                           dropout=arch['dropout'], feat=feat, aux=0)
            warm_start(m, str(apath), names)
            torch.save({'model': m.state_dict(), 'names': names, 'feat': feat,
                        'arch': arch, 'aux': 0, 'kind': 'ctc_holistic',
                        'config': {'seed': seed, 'from': str(apath)}}, align_exp)
        apath = align_exp
        import hashlib
        pool_key = hashlib.sha256(json.dumps(
            [c['id'] for c in train_clips], ensure_ascii=False).encode()).hexdigest()[:16]
        pool_path = Path(a.work) / f'fold{fold}_pool_{pool_key}.npz'
        if pool_path.exists():
            with np.load(pool_path, allow_pickle=True) as d:
                seg_pool = {int(k): list(d[k]) for k in d.files}
        else:
            seg_pool = build_pool(align_clips(apath, train_clips, dev))
            np.savez(pool_path, **{str(k): np.array(v, dtype=object)
                                   for k, v in seg_pool.items()})
        print(f'  пересборка фраз: слов в пуле {len(seg_pool)}, '
              f'отрезков {sum(len(v) for v in seg_pool.values())}', flush=True)

    # Докачка на уровне эпох: с --resume продолжаем с last-чекпойнта фолда,
    # если конфиг совпал; без --resume обучение начинается с нуля.
    last_path = Path(a.work) / f'fold{fold}_last.pt'
    start_ep = 1
    if a.resume and last_path.exists():
        st = torch.load(last_path, map_location='cpu', weights_only=False)
        want = {'epochs': a.epochs, 'steps': a.steps, 'bs': a.bs, 'lr': a.lr,
                'seed': a.seed, 'aug': a.aug, 'recombine_frac': a.recombine_frac,
                'freeze_encoder': a.freeze_encoder, 'train_clips': len(train_clips),
                'classes': len(names)}
        if st.get('resume_cfg') == want:
            model.load_state_dict(st['model'])
            opt.load_state_dict(st['opt'])
            sched.load_state_dict(st['sched'])
            start_ep = st['epoch'] + 1
            print(f'  фолд {fold}: докачка с эпохи {start_ep}', flush=True)
        else:
            print(f'  фолд {fold}: last-чекпойнт от другого конфига — с нуля', flush=True)
    for ep in range(start_ep, a.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(a.steps):
            streams, labels = [], []
            for _ in range(a.bs):
                s, lab = mix_phrase(seqs_by_class, classes, corpus, pool, rng,
                                    corpus_frac, phrase_frac, a.single_frac,
                                    aug=augment_hard if a.aug == 'hard' else augment,
                                    pool=seg_pool, recombine_frac=a.recombine_frac)
                streams.append(s)
                labels.append(lab)
            x, lens = batch_features(streams, feat)
            if a.specaug:
                x = spec_augment(x, lens.numpy(), rng)
            targets = torch.tensor([c + 1 for lab in labels for c in lab], dtype=torch.long)
            tlens = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
            logits = model(x.to(dev), lens.to(dev))
            loss = ctc(logits.log_softmax(-1).transpose(0, 1), targets.to(dev),
                       torch.from_numpy((lens.numpy() + 1) // 2).to(dev), tlens.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss)
        if ep % a.eval_every == 0 or ep == a.epochs:
            print(f'  фолд {fold} ep {ep:3d} loss {total / a.steps:.3f}', flush=True)
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'epoch': ep,
                    'resume_cfg': {'epochs': a.epochs, 'steps': a.steps,
                                   'bs': a.bs, 'lr': a.lr, 'seed': a.seed,
                                   'aug': a.aug, 'recombine_frac': a.recombine_frac,
                                   'freeze_encoder': a.freeze_encoder,
                                   'train_clips': len(train_clips),
                                   'classes': len(names)}}, last_path)
    return model


def beam_rows(path, clips, nbest=20, lm_weight=0.0, word_bonus=3.0, beam=24):
    """Production-декодер по каждому клипу: глоссы, уверенность, цепочки n-best."""
    from dialogue_translator import Recognizer
    rec = Recognizer(str(path), decoder='beam', beam=beam, lm_weight=lm_weight,
                     word_bonus=word_bonus, smooth=1, nbest=nbest)
    rows = []
    for c in clips:
        glosses, chains = rec.decode(c['span'])
        pred = [g for g, _ in glosses]
        rows.append({'id': c['id'], 'phrase': c['phrase'], 'canon': c['canon'],
                     'pred': pred, 'conf': [round(float(v), 3) for _, v in glosses],
                     'glosses': len(pred), 'chains': chains[:5] if chains else None,
                     'hits': hits(pred, c['canon'])[0],
                     'hits_strict': hits(pred, c['phrase'])[0],
                     'total': len(c['words']), 'seen_hits': 0, 'seen_total': 0})
    return rows


def beam_rows_resume(path, clips, state_path, **kw):
    """beam_rows с инкрементальным кэшем: убитый вызов продолжается с недекодированных."""
    state_path = Path(state_path)
    done_rows = {}
    if state_path.exists():
        try:
            done_rows = {r['id']: r for r in
                         json.loads(state_path.read_text(encoding='utf-8'))}
        except json.JSONDecodeError:
            done_rows = {}
    from dialogue_translator import Recognizer
    rec = Recognizer(str(path), decoder='beam', beam=kw.get('beam', 24),
                     lm_weight=kw.get('lm_weight', 0.0),
                     word_bonus=kw.get('word_bonus', 3.0), smooth=1,
                     nbest=kw.get('nbest', 20))
    rows = []
    n_new = 0
    for c in clips:
        if c['id'] in done_rows:
            rows.append(done_rows[c['id']])
            continue
        glosses, chains = rec.decode(c['span'])
        pred = [g for g, _ in glosses]
        rows.append({'id': c['id'], 'phrase': c['phrase'], 'canon': c['canon'],
                     'pred': pred, 'conf': [round(float(v), 3) for _, v in glosses],
                     'glosses': len(pred), 'chains': chains[:5] if chains else None,
                     'hits': hits(pred, c['canon'])[0],
                     'hits_strict': hits(pred, c['phrase'])[0],
                     'total': len(c['words']), 'seen_hits': 0, 'seen_total': 0})
        done_rows[c['id']] = rows[-1]
        n_new += 1
        if n_new % 10 == 0:
            state_path.write_text(json.dumps(list(done_rows.values()),
                                             ensure_ascii=False), encoding='utf-8')
    state_path.write_text(json.dumps(list(done_rows.values()),
                                     ensure_ascii=False), encoding='utf-8')
    return rows


def translate(rows):
    """Перевод и оценка смысла: пустой top-1 не идёт в LLM (B50), meaning=0."""
    tr = Translator()
    for r in rows:
        glosses = list(zip(r['pred'], r['conf']))
        text = translate_in_context(tr, glosses, [], SYSTEM, chains=r.pop('chains', None))
        r['translation'] = text
        r['meaning'] = tr.judge(text, r['phrase']) if text else 0
        r['chrf'] = round(chrf(text, r['phrase']), 3)
    return tr.cost


def attach_seen(rows, clips, vocab):
    """Дополнить beam-строки долей слов, которые были в обучении этого фолда."""
    by_id = {c['id']: c for c in clips}
    for r in rows:
        c = by_id[r['id']]
        m = matched(r['pred'], c['canon'])
        r['seen_hits'] = sum(v for w, v in m.items() if w in vocab)
        r['seen_total'] = sum(w in vocab for w in c['canon_words'])


def run(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    inv = json.loads(Path(a.inventory).read_text(encoding='utf-8'))
    plain = [w for p in inv['phrases'] if p.get('duration') for w in words(p['phrase'])]
    canon = canon_map(plain, {n.lower() for n in ck['names']}) if a.canon else {}
    all_words = [canon.get(w, w) for w in plain]
    print(f'канонизация: {sum(1 for w in plain if canon.get(w, w) != w)} из {len(plain)} '
          f'токенов → обученный класс', flush=True)
    expanded = expand_checkpoint(ck, all_words)
    names, arch, feat = list(expanded['names']), ck['arch'], ck['feat']
    init_path = work / 'init_expanded.pt'
    torch.save(expanded, init_path)
    print(f'классы {len(ck["names"])} → {len(names)} (+{expanded["expanded_added"]}); '
          f'слов фраз {len(set(all_words))}', flush=True)

    clips, skipped = load_phrase_clips(a.cache, a.inventory, names, canon)
    a.dup_ids = (set(json.loads(Path(a.dup_ids).read_text(encoding='utf-8')))
                 if a.dup_ids else set())
    a.train_ids = (set(json.loads(Path(a.train_ids).read_text(encoding='utf-8')))
                   if a.train_ids else None)
    if a.limit:
        clips = clips[:a.limit]
    if len(clips) < a.folds:
        sys.exit(f'клипов {len(clips)} меньше числа фолдов {a.folds}')
    print(f'клипов {len(clips)}, пропущено {len(skipped)}; фолдов {a.folds}', flush=True)

    d = prepare_slovo(a.data)
    seqs_by_class, classes, te_by_class, te_classes = (
        d['seqs_by_class'], d['classes'], d['te_by_class'], d['te_classes'])
    index = {n.lower(): i for i, n in enumerate(names)}
    try:
        corpus = corpus_phrases(index, 'train')
        # поток корпуса собирается из Slovo-последовательностей: классы без
        # последовательностей (новые слова вне словаря Slovo) туда не идут
        before = len(corpus)
        corpus = [p for p in corpus if all(c in seqs_by_class for c in p)]
        print(f'корпус: {before} цепочек, после фильтра классов {len(corpus)}', flush=True)
    except Exception as e:  # noqa: BLE001 — внешний кэш HF: без корпуса остаётся Slovo
        print(f'корпус недоступен ({e}) — без потоков корпуса', flush=True)
        corpus = []
    print(f'Slovo train {d["n_train"]}, test {d["n_test"]}, корпус {len(corpus)} фраз',
          flush=True)

    prog_path = work / 'progress.json'
    progress = {'meta': None, 'base_beam': None, 'folds': {}}
    if (a.resume or a.train_all) and prog_path.exists():
        try:
            loaded = json.loads(prog_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            loaded = {}
        meta = loaded.get('meta')
        want = {'seed': a.seed, 'folds': a.folds, 'clips': len(clips),
                'classes': len(names)}
        if meta == want:
            progress.update({k: v for k, v in loaded.items() if k in progress})
            print(f'прогресс: фолдов готово {len(progress["folds"])}', flush=True)
        else:
            print('прогресс из другого запуска — игнорируется', flush=True)
    progress['meta'] = {'seed': a.seed, 'folds': a.folds, 'clips': len(clips),
                        'classes': len(names)}

    def save_progress():
        prog_path.write_text(json.dumps(progress, ensure_ascii=False), encoding='utf-8')

    base_path = work / 'init_expanded_eval.pt'
    torch.save(expanded, base_path)
    if progress['base_beam'] is None:
        base = beam_rows_resume(base_path, clips, work / 'base_rows.json',
                                lm_weight=a.lm_weight,
                                word_bonus=a.word_bonus, beam=a.beam)
        progress['base_beam'] = summarize(base)
        save_progress()
    base_sum = progress['base_beam']
    print('база (без обучения):', json.dumps(base_sum, ensure_ascii=False), flush=True)

    parts = folds(len(clips), a.folds, a.seed)
    t0 = time.time()
    if a.train_all:
        # production-чекпойнт: все разрешённые клипы в обучении, CV-метрик нет (B54),
        # поэтому здесь только Slovo clean как контроль забывания
        prod_clips = training_clips(clips, [], a.train_ids)
        counts = production_counts(clips, prod_clips)
        model = train_fold(names, str(init_path), arch, feat, prod_clips,
                           seqs_by_class, classes, corpus, a, 0, dev)
        prod = Path(a.work) / 'prod.pt'
        torch.save({'model': model.state_dict(), 'names': names, 'feat': feat,
                    'arch': arch, 'aux': 0, 'kind': 'ctc_holistic',
                    'config': {'init': a.init, 'epochs': a.epochs, 'steps': a.steps,
                               'bs': a.bs, 'lr': a.lr, 'seed': a.seed, 'aug': a.aug,
                               'freeze_encoder': a.freeze_encoder, 'canon': a.canon,
                               'train_clips': counts['train_clips'],
                               'held_clips': counts['held_clips']}}, prod)
        slovo = evaluate(model, te_by_class, te_classes, np.random.default_rng(1234),
                         n=200, dev=dev)
        result = {'tool': 'finetune_ctc_phrases', 'date': time.strftime('%Y-%m-%d'),
                  'init': Path(a.init).name, 'init_sha256': file_sha256(a.init),
                  'expanded_classes': len(names), 'expanded_added': expanded['expanded_added'],
                  'train_all': True, **counts, 'skipped': skipped,
                  'seconds': round(time.time() - t0, 1), 'base_beam': base_sum,
                  'slovo_clean': {k: round(slovo[k], 3) for k in
                                  ('gloss_acc', 'exact_acc', 'edit_rate')},
                  'prod': str(prod), 'prod_sha256': file_sha256(prod),
                  'canon_mapped': len(canon),
                  'config': {'epochs': a.epochs, 'steps': a.steps, 'bs': a.bs, 'lr': a.lr,
                             'corpus_frac': a.corpus_frac, 'phrase_frac': a.phrase_frac,
                             'single_frac': a.single_frac, 'aug': a.aug,
                             'freeze_encoder': a.freeze_encoder, 'canon': a.canon,
                             'beam': a.beam, 'lm_weight': a.lm_weight,
                             'word_bonus': a.word_bonus, 'nbest': 20}}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                               encoding='utf-8')
        manifest = {'schema_version': 1, 'kind': 'ctc_holistic_phrases_prod',
                    'init': a.init, 'init_sha256': file_sha256(a.init),
                    'expanded_classes': len(names), 'canon': a.canon,
                    'train_clips': counts['train_clips'], 'git_commit': git_commit(),
                    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'checkpoints': [result['prod_sha256']]}
        Path(a.out).with_suffix('.manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
        print(json.dumps(result['slovo_clean'], ensure_ascii=False), flush=True)
        print(f'→ {a.out} (+{prod})', flush=True)
        return result
    for k, held_idx in enumerate(parts):
        if a.only_fold >= 0 and k != a.only_fold:
            continue
        key = str(k)
        if a.resume and key in progress['folds'] and (work / f'fold{k}.pt').exists():
            print(f'фолд {k}: есть в прогрессе — пропуск обучения', flush=True)
            continue
        held = [clips[i] for i in held_idx]
        train = training_clips(clips, held_idx, a.train_ids)
        vocab = {w for c in train for w in c['canon_words']}
        model = train_fold(names, str(init_path), arch, feat, train, seqs_by_class, classes,
                           corpus, a, k, dev)
        path = work / f'fold{k}.pt'
        torch.save({'model': model.state_dict(), 'names': names, 'feat': feat,
                    'arch': arch, 'aux': 0, 'kind': 'ctc_holistic',
                    'config': {'init': a.init, 'fold': k, 'epochs': a.epochs,
                               'steps': a.steps, 'bs': a.bs, 'lr': a.lr, 'seed': a.seed,
                               'aug': a.aug, 'freeze_encoder': a.freeze_encoder,
                               'train_clips': len(train), 'held_clips': len(held)}}, path)
        rows = beam_rows_resume(path, held, work / f'fold{k}_rows.json',
                                lm_weight=a.lm_weight,
                                word_bonus=a.word_bonus, beam=a.beam)
        (work / f'fold{k}_last.pt').unlink(missing_ok=True)
        attach_seen(rows, held, vocab)
        tr_g = summarize(greedy_rows(model, train[:20], names, vocab, dev))
        held_g = summarize(greedy_rows(model, held, names, vocab, dev))
        slovo = evaluate(model, te_by_class, te_classes, np.random.default_rng(1234),
                         n=30, dev=dev)
        progress['folds'][key] = {
            'fold_report': {'fold': k, 'train_clips': len(train),
                            'held_clips': len(held),
                            'beam': summarize(rows), 'train_greedy': tr_g,
                            'held_greedy': held_g,
                            'slovo_clean': {kk: round(slovo[kk], 3)
                                            for kk in ('gloss_acc', 'exact_acc', 'edit_rate')},
                            'checkpoint_sha256': file_sha256(path)},
            'rows': rows}
        save_progress()
        print(f'фолд {k}: beam {json.dumps(summarize(rows), ensure_ascii=False)} '
              f'[{time.time() - t0:.0f} с]', flush=True)

    missing = [k for k in range(a.folds) if str(k) not in progress['folds']]
    fold_report = [progress['folds'][str(k)]['fold_report'] for k in range(a.folds)
                   if str(k) in progress['folds']]
    held_rows = [r for k in range(a.folds) if str(k) in progress['folds']
                 for r in progress['folds'][str(k)]['rows']]
    if missing:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {'partial': True, 'missing_folds': missing, 'fold_report': fold_report},
            ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'не все фолды готовы, не хватает {missing} — → {a.out} (partial)',
              flush=True)
        return {'partial': True, 'missing_folds': missing}
    cost = 0.0 if a.no_llm else translate(held_rows)
    judged = [r for r in held_rows if r.get('meaning') is not None]
    final = summarize(held_rows)
    if judged:
        final.update({
            'meaning_ge1': round(sum(r['meaning'] >= 1 for r in judged) / len(judged), 3),
            'meaning_2': round(sum(r['meaning'] == 2 for r in judged) / len(judged), 3),
            'chrf': round(float(np.mean([r.get('chrf', 0.0) for r in held_rows])), 3),
            'cost_usd': round(cost, 4),
        })
    result = {'tool': 'finetune_ctc_phrases', 'date': time.strftime('%Y-%m-%d'),
              'init': Path(a.init).name, 'init_sha256': file_sha256(a.init),
              'expanded_classes': len(names), 'expanded_added': expanded['expanded_added'],
              'folds': a.folds, 'seed': a.seed,
              'config': {'epochs': a.epochs, 'steps': a.steps, 'bs': a.bs, 'lr': a.lr,
                         'corpus_frac': a.corpus_frac, 'phrase_frac': a.phrase_frac,
                         'single_frac': a.single_frac, 'aug': a.aug,
                         'recombine_frac': a.recombine_frac,
                         'freeze_encoder': a.freeze_encoder, 'beam': a.beam,
                         'lm_weight': a.lm_weight, 'word_bonus': a.word_bonus,
                         'nbest': 20},
              'clips': len(clips), 'skipped': skipped, 'seconds': round(time.time() - t0, 1),
              'base_beam': base_sum, 'fold_report': fold_report, 'held_beam': final,
              'rows': held_rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    manifest = {'schema_version': 1, 'kind': 'ctc_holistic_phrases', 'init': a.init,
                'init_sha256': file_sha256(a.init), 'expanded_classes': len(names),
                'folds': a.folds, 'seed': a.seed, 'git_commit': git_commit(),
                'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'checkpoints': [f['checkpoint_sha256'] for f in fold_report]}
    Path(a.out).with_suffix('.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(final, ensure_ascii=False), flush=True)
    print(f'→ {a.out}', flush=True)
    return result


def selftest():
    flat = sorted(i for f in folds(10, 5, 0) for i in f)
    assert flat == list(range(10)) and all(len(f) == 2 for f in folds(10, 5, 0))
    assert folds(10, 5, 0) == folds(10, 5, 0)
    assert sorted(i for f in folds(7, 3, 1) for i in f) == list(range(7))
    assert matched(['мать', 'я', 'любить'], 'Мама я тебя очень люблю') == Counter({'я': 1})
    assert matched(['там/туда'], 'туда') == Counter({'туда': 1})
    names = ['мать', 'я']
    phrases = [{'id': 'a', 'phrase': 'Мать я', 'duration': 2.0},
               {'id': 'b', 'phrase': 'Мать', 'duration': 2.0}]
    frames = np.full((20, 59, 3), 0.5, np.float32)
    path = Path('reports/evidence/_phrase_fold_selftest')
    path.mkdir(parents=True, exist_ok=True)
    try:
        for p in phrases:
            np.savez_compressed(path / f"{p['id']}.npz", frames=frames, fps=25.0,
                                phrase=p['phrase'])
        (path / 'inv.json').write_text(json.dumps({'phrases': phrases}), encoding='utf-8')
        clips, skipped = load_phrase_clips(path, path / 'inv.json', names)
        assert len(clips) == 2 and not skipped, skipped
        assert clips[0]['labels'] == [0, 1] and clips[1]['labels'] == [0]
        assert clips[0]['used'] == [0, 20]
        # меток больше, чем выходных шагов CTC: такой клип нельзя учить
        short = np.full((4, 59, 3), 0.5, np.float32)
        np.savez_compressed(path / 'b.npz', frames=short, fps=25.0, phrase='Мать я я')
        inv2 = {'phrases': [phrases[0], {'id': 'b', 'phrase': 'Мать я я', 'duration': 2.0}]}
        (path / 'inv2.json').write_text(json.dumps(inv2), encoding='utf-8')
        clips, skipped = load_phrase_clips(path, path / 'inv2.json', names)
        assert len(clips) == 1 and skipped[0]['id'] == 'b', skipped
    finally:
        for f in path.glob('*'):
            f.unlink()
        path.rmdir()
    rng = np.random.default_rng(0)
    span = np.zeros((10, 59, 3), np.float32)
    s, lab = mix_phrase({}, [], [], [(span, [0, 1])], rng, 0.0, 1.0, 0.0)
    assert lab == [0, 1] and s.shape[1:] == (59, 3)
    rows = [{'glosses': 0, 'hits': 0, 'total': 2, 'seen_hits': 0, 'seen_total': 1},
            {'glosses': 2, 'hits': 1, 'total': 2, 'seen_hits': 1, 'seen_total': 2}]
    sm = summarize(rows)
    assert sm['word_hits'] == 1 and sm['word_tokens'] == 4 and sm['empty'] == 0.5
    assert sm['seen_recall'] == round(1 / 3, 3)
    # выравнивание: два слова в очевидных участках, между ними blank
    logp = np.full((12, 3), -5.0)
    logp[:4, 1] = 0.0
    logp[4:6, 0] = 0.0
    logp[6:10, 2] = 0.0
    logp[10:, 0] = 0.0
    spans = ctc_align(logp, [0, 1])
    assert [s[0] for s in spans] == [0, 1], spans
    assert spans[0][1:] == (0, 4) and spans[1][1:] == (6, 10), spans
    # пересборка: метки по числу отрезков, кадры = отрезки + паузы
    pool = {0: [(np.zeros((8, 59, 3), np.float32), 'a')],
            1: [(np.ones((8, 59, 3), np.float32), 'b')]}
    stream, lab = recombine(pool, np.random.default_rng(0), 2, 2)
    assert len(lab) == 2 and stream.shape[1:] == (59, 3)
    assert 6 + 6 + 2 <= len(stream) <= 6 + 6 + 4, len(stream)
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/slovo/slovo_holistic.npz')
    ap.add_argument('--inventory', default='reports/evidence/rutube_channel_inventory.json')
    ap.add_argument('--cache', default='D:/hfdata/rutube_phrases/landmarks')
    ap.add_argument('--init', default='models/slovo_ctc_motion_head_all.pt')
    ap.add_argument('--work', default='D:/hfdata/phrase_folds')
    ap.add_argument('--out', default='reports/evidence/phrase_stage20.json')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--corpus-frac', type=float, default=0.30)
    ap.add_argument('--phrase-frac', type=float, default=0.45)
    ap.add_argument('--single-frac', type=float, default=0.10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--specaug', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--canon', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--train-all', action='store_true',
                    help='production-чекпойнт на всех клипах; CV-метрик не будет (B54)')
    ap.add_argument('--aug', choices=('base', 'hard'), default='base')
    ap.add_argument('--recombine-frac', type=float, default=0.0)
    ap.add_argument('--align-from', default='D:/hfdata/phrase_folds/fold{fold}.pt')
    ap.add_argument('--freeze-encoder', action='store_true')
    ap.add_argument('--lm-weight', type=float, default=0.0)
    ap.add_argument('--word-bonus', type=float, default=3.0)
    ap.add_argument('--beam', type=int, default=24)
    ap.add_argument('--no-llm', action='store_true')
    ap.add_argument('--resume', action='store_true',
                    help='не переобучать фолд, если чекпойнт и прогресс есть; '
                         'прогресс хранится в <work>/progress.json')
    ap.add_argument('--dup-times', type=int, default=0,
                    help='сколько раз продублировать клипы из --dup-ids в пуле фраз')
    ap.add_argument('--dup-ids', default='',
                    help='JSON-файл со списком id клипов для оверсэмплинга')
    ap.add_argument('--train-ids', default='',
                    help='JSON со списком разрешённых id только для train; held не фильтруется')
    ap.add_argument('--only-fold', type=int, default=-1,
                    help='обработать только этот фолд (вместе с --resume)')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)


if __name__ == '__main__':
    main()
