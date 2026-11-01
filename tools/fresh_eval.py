"""Честный fresh-тест production-цепочки на видео ВНЕ обучения (B54).

Зачем: 67 фразовых клипов израсходованы как train+CV, у адаптированных моделей нет
независимой фразовой метрики. Ролики канала вне 67 клипов с фразой в заголовке
(например «Жестовый язык. Фразы. Мама я тебя очень люблю!!!») дают честный тест:
ролик не участвовал ни в обучении, ни в CV.

Для каждого видео: ландмарки с кэшем → сплены по паузам → beam-декод каждой модели
→ все цепочки в LLM → перевод; судья 0/1/2 против фразы из заголовка. Модели
задаются списком «имя=путь» и сравниваются на одних сегментах.

  python tools/fresh_eval.py --video D:/hfdata/demo_frases/mama_love.mp4 \
      --phrase "Мама я тебя очень люблю" \
      --models prod=D:/hfdata/phrase20/prod/prod.pt,dict=D:/hfdata/dict_folds/prod_dict.pt
  python tools/fresh_eval.py --selftest
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
from dialogue_translator import Recognizer  # noqa: E402
from eval_phrase_clips import load_frames  # noqa: E402
from gloss_to_text import SYSTEM, Translator, chrf  # noqa: E402
from sign_utils import split_utterances  # noqa: E402


def parse_models(spec):
    out = []
    for part in spec.split(','):
        name, _, path = part.partition('=')
        if not name or not path:
            raise ValueError(f'формат --models имя=путь: {part!r}')
        out.append((name, path))
    return out


def run(a):
    frames, fps, state = load_frames(a.video, a.cache, a.phrase)
    spans = split_utterances(frames, fps, pause=0.7, min_len=0.4, max_len=15.0)
    spans = spans or [(0, len(frames))]
    print(f'видео {Path(a.video).name}: {len(frames)} кадров @ {fps:.0f} fps '
          f'({state}), спанов с руками: {len(spans)}', flush=True)
    tr = Translator()
    rows, t0 = [], time.time()
    for name, path in parse_models(a.models):
        rec = Recognizer(path, decoder='beam', beam=24, lm_weight=0.0,
                         word_bonus=a.word_bonus, smooth=1, nbest=20)
        for si, (x, y) in enumerate(spans):
            glosses, chains = rec.decode(frames[x:y])
            pred = [g for g, _ in glosses]
            text = translate_in_context(tr, glosses, [], SYSTEM,
                                        chains=chains[:5]) if pred else ''
            meaning = tr.judge(text, a.phrase) if text else 0
            rows.append({'model': name, 'span': [int(x), int(y)],
                         'pred': pred, 'translation': text, 'meaning': meaning,
                         'chrf': round(chrf(text, a.phrase), 3)})
            print(f'{name} #{si + 1}: {pred} → {text!r} ({meaning})', flush=True)
    summary = {}
    for name, _ in parse_models(a.models):
        rs = [r for r in rows if r['model'] == name]
        best = max(rs, key=lambda r: (r['meaning'], r['chrf']))
        judged = [r for r in rs if r['translation']]
        summary[name] = {
            'segments': len(rs),
            'meaning_ge1': round(sum(r['meaning'] >= 1 for r in judged)
                                 / max(len(judged), 1), 3),
            'best_meaning': best['meaning'], 'best_translation': best['translation'],
            'chrf_mean': round(float(sum(r['chrf'] for r in rs)) / max(len(rs), 1), 2),
            'chrf_best': best['chrf'],
        }
    out = {'tool': 'fresh_eval', 'date': time.strftime('%Y-%m-%d'),
           'video': str(a.video), 'phrase': a.phrase, 'fps': fps,
           'summary': summary, 'cost_usd': round(tr.cost, 4),
           'seconds': round(time.time() - t0, 1), 'rows': rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f'→ {a.out}', flush=True)


def selftest():
    assert parse_models('a=/x.pt,b=/y.pt') == [('a', '/x.pt'), ('b', '/y.pt')]
    try:
        parse_models('bad')
        raise AssertionError('должен падать без «=»')
    except ValueError:
        pass
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='')
    ap.add_argument('--phrase', default='', help='эталонная фраза (из заголовка)')
    ap.add_argument('--models', default='', help='имя=путь через запятую')
    ap.add_argument('--cache', default='D:/hfdata/fresh_landmarks')
    ap.add_argument('--out', default='reports/evidence/fresh_eval.json')
    ap.add_argument('--word-bonus', type=float, default=3.0)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.video and a.phrase and a.models):
        ap.error('нужны --video, --phrase и --models')
    else:
        run(a)


if __name__ == '__main__':
    main()