"""Фразовый замер на живой съёмке: 67 клипов Rutube с точной фразой в заголовке.

Для каждого клипа: MediaPipe Holistic → [T, 59, 3] (кэш на D:) → CTC motion beam 24 /
nbest 20 → все цепочки в LLM → перевод; судья ставит 0/1/2. Дополнительно считается
точное совпадение слов перевода с эталоном и покрытие эталона словарём модели.

  python tools/eval_phrase_clips.py --no-llm --limit 5
  python tools/eval_phrase_clips.py --selftest
  python tools/eval_phrase_clips.py

Видео и ландмарки остаются на D:, в репозиторий идёт только JSON замера.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dialogue_translator import Recognizer, frame_of  # noqa: E402
from dialogue_state import translate_in_context  # noqa: E402
from gloss_to_text import SYSTEM, Translator, chrf  # noqa: E402
from sign_utils import split_utterances  # noqa: E402


def words(text):
    """Сравнимое представление фразы: нижний регистр, ё→е, без пунктуации."""
    return re.findall(r'[а-яa-z0-9]+', text.lower().replace('ё', 'е'))


def hits(pred, ref):
    """Точное совпадение слов как мешок: (попадания, полнота, точность)."""
    pred = [pred] if isinstance(pred, str) else pred
    ref = [ref] if isinstance(ref, str) else ref
    p, r = Counter(words(' '.join(pred))), Counter(words(' '.join(ref)))
    n = sum((p & r).values())
    return n, (n / sum(r.values()) if r else 0.0), (n / sum(p.values()) if p else 0.0)


def landmarks(path):
    """Видео → [T, 59, 3] по каждому кадру (как в живом режиме) и fps."""
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    with mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.3, min_tracking_confidence=0.3) as h:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(frame_of(h.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
    cap.release()
    return np.stack(frames).astype(np.float32), fps


def load_frames(path, cache, phrase):
    """Кэш ландмарков по id клипа; повторный прогон не гоняет MediaPipe."""
    out = Path(cache) / f'{Path(path).stem}.npz'
    if out.exists():
        with np.load(out) as d:
            return d['frames'], float(d['fps']), 'cached'
    frames, fps = landmarks(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=frames, fps=fps, phrase=phrase)
    return frames, fps, 'fresh'


def hands_ratio(frames):
    hands = frames[:, :42].reshape(len(frames), 2, 21, 3)
    return float((~np.isnan(hands[:, :, 0, 0]).all(1)).mean())


def score(summary):
    n = max(len(summary), 1)
    judged = [r for r in summary if r.get('meaning') is not None]
    return {
        'clips': len(summary),
        'failed': sum(1 for r in summary if r.get('error')),
        'hands_ratio': round(float(np.mean([r.get('hands_ratio', 0.0) for r in summary])), 3),
        'spans_mean': round(float(np.mean([r.get('spans', 0) for r in summary])), 2),
        'glosses_mean': round(float(np.mean([r.get('glosses', 0) for r in summary])), 2),
        'word_recall': round(float(np.mean([r.get('word_recall', 0.0) for r in summary])), 3),
        'word_precision': round(float(np.mean([r.get('word_precision', 0.0) for r in summary])), 3),
        'exact_translation': round(sum(bool(r.get('exact')) for r in summary) / n, 3),
        'meaning_missing': len(summary) - len(judged),
        'meaning_ge1': round(sum(r['meaning'] >= 1 for r in judged) / max(len(judged), 1), 3),
        'meaning_2': round(sum(r['meaning'] == 2 for r in judged) / max(len(judged), 1), 3),
        'chrf': round(float(np.mean([r.get('chrf', 0.0) for r in summary])), 3),
        'vocab_coverage': round(float(np.mean([r.get('vocab_coverage', 0.0) for r in summary])), 3),
    }


def run(media, inventory, cache, model, out, limit=0, no_llm=False):
    inv = json.loads(Path(inventory).read_text(encoding='utf-8'))
    clips = [p for p in inv['phrases'] if p.get('duration')]
    if limit:
        clips = clips[:limit]
    print(f'клипов: {len(clips)}', flush=True)

    rec = Recognizer(str(model), decoder='beam', beam=24, lm_weight=0.1,
                     word_bonus=2.0, smooth=1, nbest=20)
    vocab = {n.lower().replace('ё', 'е') for n in rec.names}
    tr = Translator() if not no_llm else None
    started = time.time()
    rows = []
    for i, p in enumerate(clips, 1):
        path = Path(media) / f"{p['id']}.mp4"
        row = {'id': p['id'], 'phrase': p['phrase'], 'duration': p['duration']}
        if not path.exists():
            row['error'] = 'нет файла'
            rows.append(row)
            continue
        try:
            frames, fps, state = load_frames(path, cache, p['phrase'])
            spans = split_utterances(frames, fps, pause=0.7, min_len=0.4, max_len=15.0)
            a, b = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(frames))
            glosses, chains = rec.decode(frames[a:b])
            pred = [g for g, _ in glosses]
            n_hit, recall, precision = hits(pred, p['phrase'])
            ref = words(p['phrase'])
            row.update({
                'frames': int(len(frames)), 'used': [int(a), int(b)], 'spans': len(spans),
                'cache': state, 'hands_ratio': round(hands_ratio(frames), 3),
                'glosses': len(pred), 'pred': pred,
                'conf': [round(float(c), 3) for _, c in glosses],
                'word_hits': n_hit, 'word_total': len(ref),
                'word_recall': round(recall, 3), 'word_precision': round(precision, 3),
                'vocab_coverage': round(sum(w in vocab for w in ref) / max(len(ref), 1), 3),
                'vocab_missing': [w for w in ref if w not in vocab],
                'chains': chains[:5] if chains else None,
            })
            if tr:
                text = translate_in_context(tr, glosses, [], SYSTEM, chains=chains)
                row['translation'] = text
                row['meaning'] = tr.judge(text, p['phrase'])
                row['chrf'] = round(chrf(text, p['phrase']), 3)
                row['exact'] = words(text) == ref
        except Exception as e:  # без ретраев: один сбой остаётся в отчёте
            row['error'] = f'{type(e).__name__}: {e}'
        rows.append(row)
        print(f"{i}/{len(clips)} {p['id'][:8]} {p['phrase']!r}: "
              f"{row.get('pred', row.get('error'))}", flush=True)

    result = {'media': str(media), 'model': Path(model).name,
              'config': {'beam': 24, 'nbest': 20, 'lm_weight': 0.1, 'word_bonus': 2.0,
                         'fps': 'each frame (as live)', 'split': 'longest hand span'},
              'date': time.strftime('%Y-%m-%d'), 'seconds': round(time.time() - started, 1),
              'cost_usd': round(tr.cost, 4) if tr else 0.0,
              'summary': score(rows), 'clips': rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)
    print(f'→ {out}', flush=True)
    return result


def selftest():
    assert words('Мама, я тебя ОЧЕНЬ люблю!') == ['мама', 'я', 'тебя', 'очень', 'люблю']
    assert hits(['мать', 'я', 'любить'], 'Мама я тебя очень люблю') == (1, 0.2, 1 / 3)
    assert hits([], 'мама') == (0, 0.0, 0.0)
    rows = [{'hands_ratio': 1.0, 'spans': 1, 'glosses': 3, 'word_recall': 0.5,
             'word_precision': 1.0, 'vocab_coverage': 0.5, 'meaning': 1, 'chrf': 40.0}]
    s = score(rows)
    assert s['meaning_ge1'] == 1.0 and s['meaning_2'] == 0.0 and s['clips'] == 1
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--media', default='D:/hfdata/rutube_phrases')
    ap.add_argument('--inventory',
                    default='reports/evidence/rutube_channel_inventory.json')
    ap.add_argument('--cache', default='D:/hfdata/rutube_phrases/landmarks')
    ap.add_argument('--model', default=str(ROOT / 'models' / 'slovo_ctc_motion.pt'))
    ap.add_argument('--out', default='reports/evidence/phrase_clips_eval.json')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-llm', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.media, a.inventory, a.cache, a.model, a.out, a.limit, a.no_llm)


if __name__ == '__main__':
    main()
