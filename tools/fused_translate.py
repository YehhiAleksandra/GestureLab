"""Фьюжн-переводчик: CTC (ландмарки) + easy_sign (S3D по кадрам) → русская фраза.

Зачем: CTC-модель обучена на 67 фразах одного исполнителя и теряет содержательные
слова на новых видео; easy_sign (1598 жестов РЖЯ, обучена на ~180k примеров) видит
их независимо — по видеокадрам, а не по ландмаркам. На свежем mama_love easy_sign
нашла «любовь», которую CTC пропустила. Инструмент сливает глоссы обеих моделей
по временной шкале и отдаёт объединённую последовательность в LLM-переводчик.

Вероятности easy_sign — softmax поверх выхода S3D (сырой выход — логиты, не
вероятности). Окна 2 с, шаг 0,5 с; фильтр: softmax p >= --es-min (по умолчанию
0,5) и класс не 'no*'. Подряд идущие одинаковые классы схлопываются в один жест.

Режимы:
  # свежий перевод одного видео
  python tools/fused_translate.py --video x.mp4 --phrase "..." \
      --model D:/hfdata/rsl_translator_v1/prod_v2.pt

  # честная оценка фьюжна на отложенных клипах (без LLM: recall; с LLM: смысл)
  python tools/fused_translate.py --eval-folds --work D:/hfdata/phrase_recombine \
      --videos D:/hfdata/rutube_phrases --cache D:/hfdata/rutube_phrases/landmarks

  python tools/fused_translate.py --selftest
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dialogue_state import translate_in_context  # noqa: E402
from dialogue_translator import Recognizer  # noqa: E402
from eval_phrase_clips import load_frames, words  # noqa: E402
from gloss_to_text import SYSTEM, Translator, chrf  # noqa: E402
from sign_utils import split_utterances  # noqa: E402

ES_MODEL = 'models/easy_sign/S3D.onnx'
ES_CLASSES = 'models/easy_sign/RSL_class_list.txt'


def load_es_classes(path=ES_CLASSES):
    classes = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            classes[int(parts[0])] = parts[1]
    return classes


def es_segments(video, sess=None, classes=None, window=2.0, step=0.5,
                p_min=0.5, cache_dir=None):
    """Видео → [(слово, conf, t_mid)] схлопнутых сегментов easy_sign.

    sess/classes — для повторного использования сессии ONNX в батче.
    cache_dir — json-кэш по имени файла (батч по 67 клипов идёт долго).
    """
    import cv2
    import onnxruntime as ort
    cache = Path(cache_dir) / f'{Path(video).stem}.json' if cache_dir else None
    if cache and cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    if sess is None:
        sess = ort.InferenceSession(str(ROOT / ES_MODEL))
    if classes is None:
        classes = load_es_classes()
    input_name = sess.get_inputs()[0].name
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = max(1, int(window * fps))
    stp = max(1, int(step * fps))
    rows = []
    start = 0
    while start + win <= total:
        idxs = np.linspace(start, start + win - 1, 32).astype(int)
        frames = []
        last = None
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                fr = cv2.resize(fr, (224, 224))
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                last = fr
            frames.append(last if last is not None else np.zeros((224, 224, 3), np.uint8))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        x = np.array(frames, np.float32) / 255.0
        x = np.transpose(x, (3, 0, 1, 2))[None]
        logits = sess.run(None, {input_name: x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        c = int(np.argmax(p))
        rows.append({'t': (start + win / 2) / fps, 'cls': c, 'p': float(p[c])})
        start += stp
    cap.release()
    segs, cur = [], None
    for r in rows:
        name = classes.get(r['cls'], f"unk_{r['cls']}")
        if cur and cur['w'] == name:
            cur['p'].append(r['p'])
            cur['t1'] = r['t']
        else:
            if cur:
                segs.append(cur)
            cur = {'w': name, 'p': [r['p']], 't0': r['t'], 't1': r['t']}
    if cur:
        segs.append(cur)
    out = [{'w': s['w'], 'conf': round(float(np.mean(s['p'])), 3),
            't': round((s['t0'] + s['t1']) / 2, 2)} for s in segs
           if s['w'].lower() not in ('no', 'no_event') and float(np.mean(s['p'])) >= p_min]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    return out


def merge(ctc_pred, es_segs, span_start_s, span_end_s):
    """Слияние: слова easy_sign, отсутствующие в CTC-предсказании, вставляются
    в позицию по временной шкале спана. Возвращает [(слово, conf)]."""
    base = [(w, 0.8) for w in ctc_pred]
    dur = max(span_end_s - span_start_s, 1e-6)
    known = {w.lower().replace('ё', 'е') for w in ctc_pred}
    additions = [s for s in es_segs
                 if s['w'].lower().replace('ё', 'е') not in known]
    if not additions:
        return base
    pos = {}
    for i, s in enumerate(additions):
        frac = min(max((s['t'] - span_start_s) / dur, 0.0), 0.999)
        pos[i] = int(round(frac * len(base)))
    out = list(base)
    order = sorted(range(len(additions)),
                 key=lambda i2: (pos[i2], additions[i2]['t']), reverse=True)
    for i in order:
        out.insert(pos[i], (additions[i]['w'], additions[i]['conf']))
    return out


def run_video(a):
    frames, fps, state = load_frames(a.video, a.cache, a.phrase)
    spans = split_utterances(frames, fps, pause=0.7, min_len=0.4, max_len=15.0)
    spans = spans or [(0, len(frames))]
    rec = Recognizer(a.model, decoder='beam', beam=24, lm_weight=0.0,
                     word_bonus=3.0, smooth=1, nbest=20)
    tr = Translator()
    rows = []
    for si, (x, y) in enumerate(spans):
        glosses, chains = rec.decode(frames[x:y])
        ctc_pred = [g for g, _ in glosses]
        es = es_segments(a.video, p_min=a.es_min,
                         cache_dir=a.es_cache) if si == 0 else json.loads(
            (Path(a.es_cache) / f'{Path(a.video).stem}.json').read_text(encoding='utf-8'))
        es_in = [s for s in es if x / fps - 1.0 <= s['t'] <= y / fps + 1.0]
        merged = merge(ctc_pred, es_in, x / fps, y / fps)
        text_c = translate_in_context(tr, glosses, [], SYSTEM, chains=chains[:5]) if ctc_pred else ''
        # фьюжн: без цепочек CTC — иначе LLM видит только их и слова
        # easy_sign игнорируются
        text_f = translate_in_context(tr, merged, [], SYSTEM)
        rows.append({'span': [int(x), int(y)], 'ctc_pred': ctc_pred,
                     'es_in_span': es_in, 'merged': [w for w, _ in merged],
                     'text_ctc': text_c, 'text_fused': text_f,
                     'meaning_ctc': tr.judge(text_c, a.phrase) if text_c else 0,
                     'meaning_fused': tr.judge(text_f, a.phrase) if text_f else 0,
                     'chrf_ctc': round(chrf(text_c, a.phrase), 3) if text_c else 0.0,
                     'chrf_fused': round(chrf(text_f, a.phrase), 3) if text_f else 0.0})
        print(f'#{si + 1} ctc={ctc_pred} es={[s["w"] for s in es_in]} '
              f'→ {text_f!r} (смысл {rows[-1]["meaning_fused"]})', flush=True)
    out = {'tool': 'fused_translate', 'date': time.strftime('%Y-%m-%d'),
           'video': str(a.video), 'phrase': a.phrase, 'model': a.model,
           'es_min': a.es_min, 'cost_usd': round(tr.cost, 4), 'rows': rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    print(f'→ {a.out}', flush=True)
    return out


def run_eval_folds(a):
    """Честная оценка фьюжна: отложенные клипы каждого фолда.

    Источник строк: --report (финальный отчёт CV-прогона) с восстановлением
    фолдов по seed (--seed); --work/progress.json — запасной вариант."""
    import onnxruntime as ort
    from finetune_ctc_phrases import folds as cv_folds
    inv = json.loads(Path(a.inventory).read_text(encoding='utf-8'))
    inv_ids = [p['id'] for p in inv['phrases'] if p.get('duration')]
    if Path(a.report).exists():
        rep = json.loads(Path(a.report).read_text(encoding='utf-8'))
        parts = cv_folds(len(inv_ids), a.folds, a.seed)
        fold_of = {inv_ids[i]: k for k, part in enumerate(parts) for i in part}
        rows = [dict(r, fold=fold_of.get(r['id'])) for r in rep['rows']]
        rows = [r for r in rows if r['fold'] is not None]
    else:
        work = Path(a.work)
        progress = json.loads((work / 'progress.json').read_text(encoding='utf-8'))
        rows = [dict(r, fold=int(k)) for k in sorted(progress['folds'], key=int)
                for r in progress['folds'][k]['rows']]
    sess = ort.InferenceSession(str(ROOT / ES_MODEL))
    classes = load_es_classes()
    report, tr = [], Translator()
    if rows:
        for r in rows:
            cid = r['id']
            video = Path(a.videos) / f'{cid}.mp4'
            if not video.exists():
                continue
            es = es_segments(video, sess=sess, classes=classes, p_min=a.es_min,
                             cache_dir=a.es_cache)
            frames, fps, _ = load_frames(video, a.cache, r['phrase'])
            span_start, span_end = 0.0, len(frames) / fps
            es_in = [s for s in es if span_start - 1.0 <= s['t'] <= span_end + 1.0]
            merged = merge(r['pred'], es_in, span_start, span_end)
            ref = set(words(r['phrase']))
            ctc_hits = len(ref & set(words(' '.join(r['pred']))))
            fus_hits = len(ref & set(words(' '.join(w for w, _ in merged))))
            row = {'fold': r['fold'], 'id': cid, 'phrase': r['phrase'],
                   'ctc_pred': r['pred'], 'es': es_in,
                   'merged': [w for w, _ in merged],
                   'ctc_hits': ctc_hits, 'fused_hits': fus_hits,
                   'total': len(ref)}
            if fus_hits != ctc_hits and a.llm:
                text_c = translate_in_context(tr, list(zip(r['pred'], r['conf'])), [],
                                              SYSTEM, chains=r.get('chains'))
                text_f = translate_in_context(tr, merged, [], SYSTEM)
                row.update({'text_ctc': text_c, 'text_fused': text_f,
                            'meaning_ctc': tr.judge(text_c, r['phrase']) if text_c else 0,
                            'meaning_fused': tr.judge(text_f, r['phrase']) if text_f else 0,
                            'chrf_ctc': round(chrf(text_c, r['phrase']), 3) if text_c else 0.0,
                            'chrf_fused': round(chrf(text_f, r['phrase']), 3) if text_f else 0.0})
            report.append(row)
            print(f'фолд {r["fold"]} {cid[:8]}: ctc {ctc_hits}/{len(ref)} → fused '
                  f'{fus_hits}/{len(ref)}', flush=True)
    tot_c = sum(r['ctc_hits'] for r in report)
    tot_f = sum(r['fused_hits'] for r in report)
    tot_t = sum(r['total'] for r in report)
    diff = [r for r in report if 'meaning_fused' in r]
    out = {'tool': 'fused_translate', 'mode': 'eval_folds',
           'date': time.strftime('%Y-%m-%d'), 'es_min': a.es_min,
           'clips': len(report), 'ctc_hits': tot_c, 'fused_hits': tot_f,
           'tokens': tot_t,
           'ctc_recall': round(tot_c / max(tot_t, 1), 3),
           'fused_recall': round(tot_f / max(tot_t, 1), 3),
           'llm_compared': len(diff),
           'meaning_ctc': round(sum(r['meaning_ctc'] for r in diff) / max(len(diff), 1), 3),
           'meaning_fused': round(sum(r['meaning_fused'] for r in diff) / max(len(diff), 1), 3),
           'chrf_ctc': round(float(np.mean([r['chrf_ctc'] for r in diff])), 3) if diff else None,
           'chrf_fused': round(float(np.mean([r['chrf_fused'] for r in diff])), 3) if diff else None,
           'cost_usd': round(tr.cost, 4), 'rows': report}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False),
          flush=True)
    print(f'→ {a.out}', flush=True)
    return out


def selftest():
    base = [('мать', 0.8), ('я', 0.8)]
    es = [{'w': 'любовь', 'conf': 0.7, 't': 3.0}]
    m = merge([w for w, _ in base], es, 0.0, 8.0)
    # 3 с из 8 → 37,5 % шкалы → позиция 1 из 2
    assert [w for w, _ in m] == ['мать', 'любовь', 'я'], m
    m0 = merge([w for w, _ in base], [{'w': 'x', 'conf': 0.7, 't': 9.0}], 0.0, 8.0)
    assert [w for w, _ in m0] == ['мать', 'я', 'x'], m0  # за пределами — в конец
    es2 = [{'w': 'мать', 'conf': 0.9, 't': 1.0}]
    assert merge(['мать'], es2, 0.0, 4.0) == [('мать', 0.8)], 'дубликат не добавляется'
    assert merge(['Мать'], es2, 0.0, 4.0) == [('Мать', 0.8)], 'регистр не создаёт дубликат'
    m3 = merge([], [{'w': 'a', 'conf': 0.6, 't': 1.0},
                    {'w': 'b', 'conf': 0.6, 't': 3.0}], 0.0, 4.0)
    assert [w for w, _ in m3] == ['a', 'b'], m3
    assert load_es_classes  # импортируется
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='')
    ap.add_argument('--phrase', default='')
    ap.add_argument('--model', default='D:/hfdata/rsl_translator_v1/prod_v3.pt')
    ap.add_argument('--cache', default='D:/hfdata/fresh_landmarks')
    ap.add_argument('--es-cache', default='D:/hfdata/es_cache')
    ap.add_argument('--es-min', type=float, default=0.5)
    ap.add_argument('--out', default='reports/evidence/fused_translate.json')
    ap.add_argument('--eval-folds', action='store_true')
    ap.add_argument('--report', default='reports/evidence/phrase_recombine.json',
                    help='CV-отчёт с held-out строками (восстановление фолдов по seed)')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--work', default='D:/hfdata/phrase_recombine')
    ap.add_argument('--inventory', default='reports/evidence/rutube_channel_inventory.json')
    ap.add_argument('--videos', default='D:/hfdata/rutube_phrases')
    ap.add_argument('--llm', action='store_true', help='LLM-перевод клипов с различиями')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.eval_folds:
        run_eval_folds(a)
    elif a.video and a.phrase:
        run_video(a)
    else:
        ap.error('нужны --video+--phrase или --eval-folds')


if __name__ == '__main__':
    main()
