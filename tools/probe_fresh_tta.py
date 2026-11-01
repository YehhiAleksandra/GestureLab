"""Офлайн-проверка TTA на независимом видео, без LLM и изменения модели."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dialogue_translator import Recognizer  # noqa: E402
from eval_phrase_clips import hits, load_frames, words  # noqa: E402
from rerank_eval import chain_bag, merge_bags  # noqa: E402
from sign_utils import split_utterances  # noqa: E402
from slovo_model import resample  # noqa: E402


def run(a):
    frames, fps, cache = load_frames(a.video, a.cache, a.phrase)
    spans = split_utterances(frames, fps, pause=0.7, min_len=0.4, max_len=15.0)
    start, end = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(frames))
    span = frames[start:end]
    speeds = [float(x) for x in a.tta.split(',')]
    ref = words(a.phrase)
    rows = []
    for name, path in (p.split('=', 1) for p in a.models.split(',')):
        rec = Recognizer(path, decoder='beam', beam=24, lm_weight=0.0,
                         word_bonus=3.0, smooth=1, nbest=20)
        pools = []
        for speed in speeds:
            variant = span if speed == 1.0 else resample(
                span, t=max(8, round(len(span) / speed)))
            glosses, chains = rec.decode(variant)
            pool = chains or [[g for g, _ in glosses]]
            pools.append(pool)
            top = pool[0]
            rows.append({'model': name, 'speed': speed, 'top1': top,
                         'hits': hits(top, ref)[0], 'chains': pool[:5]})
        bag = merge_bags(*[chain_bag(pool, 5) for pool in pools])
        print(f'{name}: TTA bag = {dict(bag)}', flush=True)
    summary = {
        'reference': ref,
        'baseline_hits': sum(r['hits'] for r in rows if r['speed'] == 1.0),
        'best_tta_top1_hits': max((r['hits'] for r in rows), default=0),
        'models': len({r['model'] for r in rows}),
        'speeds': speeds,
    }
    out = {'tool': 'probe_fresh_tta', 'date': time.strftime('%Y-%m-%d'),
           'video': str(a.video), 'phrase': a.phrase, 'span': [start, end],
           'cache': cache, 'summary': summary, 'rows': rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f'→ {a.out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='D:/hfdata/demo_frases/mama_love.mp4')
    ap.add_argument('--phrase', default='Мама я тебя очень люблю')
    ap.add_argument('--models', required=True, help='имя=путь,имя=путь')
    ap.add_argument('--cache', default='D:/hfdata/fresh_landmarks')
    ap.add_argument('--tta', default='1.0,0.85,1.2')
    ap.add_argument('--out', default='reports/evidence/fresh_tta_probe.json')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
