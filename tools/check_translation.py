"""Проверка перевода по эталону: ролик с известной фразой → видео с полосой
«эталон / глоссы / перевод» под кадром, скелетом поверх и озвучкой.

Для тех, кто не знает РЖЯ: эталонная фраза стоит рядом с тем, что увидела модель,
поэтому совпадение проверяется глазами, без умения читать жесты.

  python tools/check_translation.py --video D:\\hfdata\\demo_frases\\mama_love.mp4 \\
      --reference "мама я тебя очень люблю" --out D:\\hfdata\\verify\\mama_check.mp4
  python tools/check_translation.py --video D:\\hfdata\\demo_dialog.mp4 \\
      --reference "друг прийти потом | пить вода" --out D:\\hfdata\\verify\\dialog_check.mp4
  python tools/check_translation.py --selftest

Реплики в эталоне разделяются `|` и сопоставляются найденным паузам по порядку.
Видео и mp3 пишутся только на диск D: (реальные люди в репозиторий не попадают);
в репозиторий идёт лишь JSON замера.
"""
import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dialogue_state import translate_in_context  # noqa: E402
from dialogue_translator import Recognizer, TUNED_DECODER, frame_of  # noqa: E402
from eval_phrase_clips import hits  # noqa: E402
from gl_capture import fps_of, open_video  # noqa: E402
from gloss_to_text import SYSTEM, Translator, chrf  # noqa: E402
from sign_utils import split_utterances  # noqa: E402
from skeleton_overlay import draw_holistic  # noqa: E402

BAR = 190
REF_COLOR = (230, 200, 140)    # BGR: голубой — эталон
GLOSS_COLOR = (0, 255, 255)    # жёлтый — что увидела модель
TEXT_COLOR = (255, 255, 255)   # белый — перевод


def wrap(text, max_chars, lines=2):
    """Словами, не по буквам; лишнее — многоточие в последней строке."""
    out, line = [], ''
    for w in text.split():
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + ' ' + w).strip()
        else:
            out.append(line)
            line = w
    if line:
        out.append(line)
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][:max_chars - 1] + '…'
    return out


def panel(frame, rows, height=BAR):
    """Кадр + полоса под ним; rows — [(цвет, текст)], текст режется на две строки."""
    h, w = frame.shape[:2]
    bar = np.zeros((height, w, 3), np.uint8)
    scale = max(0.45, min(0.7, w / 1100))
    max_chars = max(18, int(w / (9 * scale / 0.6)))
    y = 8
    for color, text in rows:
        for line in wrap(text, max_chars):
            cv2.putText(bar, line, (10, y + int(24 * scale / 0.6)), cv2.FONT_HERSHEY_COMPLEX,
                        scale, color, 1, cv2.LINE_AA)
            y += int(26 * scale / 0.6)
        y += 4
    return cv2.vconcat([frame, bar])


def rows_for(i, spans, refs, preds, texts):
    """Строки полосы на кадре i: последняя начавшаяся реплика (видна и в паузе)."""
    cur = -1
    for k, (a, _b) in enumerate(spans):
        if i >= a:
            cur = k
    if cur < 0:
        return [(REF_COLOR, 'ждём жест…')]
    ref = refs[cur] if cur < len(refs) else ''
    return [(REF_COLOR, f'эталон: {ref}' if ref else 'эталон: —'),
            (GLOSS_COLOR, f'модель: {" ".join(preds[cur]) or "—"}'),
            (TEXT_COLOR, f'перевод: {texts[cur] or "—"}')]


def draw_source(video, silent, pause=0.7, max_len=15.0):
    """Прогон 1: ландмарки + скелет в немое видео. Возвращает (ландмарки, fps, размеры)."""
    cap = open_video(video)
    fps = fps_of(cap)
    ok, first = cap.read()
    if not ok:
        sys.exit(f'в видео нет кадров: {video}')
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                              smooth_landmarks=True)
    frames = []
    try:
        frame = first
        while ok:
            arr = frame_of(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            hands = arr[:42].reshape(2, 21, 3)
            draw_holistic(frame, hands[0], hands[1], arr[42:49], arr[49:])
            writer.write(frame)
            frames.append(arr)
            ok, frame = cap.read()
    finally:
        holistic.close()
        cap.release()
        writer.release()
    return np.stack(frames), fps, (w, h)


def edges(arr, fps):
    hands = arr[:, :42].reshape(len(arr), 2, 21, 3)
    return float((~np.isnan(hands[:, :, 0, 0]).all(1)).mean())


def _tts(text, path, voice):
    import edge_tts

    async def go():
        await edge_tts.Communicate(text, voice).save(str(path))
    asyncio.run(go())


def mux(video, voices, out, duration):
    """Голос по таймкодам поверх тишины; исходная дорожка не нужна — она музыка/шум."""
    if not voices:
        shutil.copyfile(video, out)
        return
    args = ['ffmpeg', '-y', '-i', str(video), '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono']
    for p, _ in voices:
        args += ['-i', str(p)]
    parts = [f'[{k + 2}:a]adelay={int(start * 1000)}:all=1[a{k}]'
             for k, (_p, start) in enumerate(voices)]
    mix = '[1:a]' + ''.join(f'[a{k}]' for k in range(len(voices)))
    parts.append(f'{mix}amix=inputs={len(voices) + 1}:duration=longest:normalize=0[aout]')
    args += ['-filter_complex', ';'.join(parts), '-map', '0:v', '-map', '[aout]',
             '-c:v', 'copy', '-c:a', 'aac', '-t', f'{duration:.3f}', str(out)]
    subprocess.run(args, check=True, capture_output=True)


def run(video, reference, out, model, beam=24, nbest=20, no_tts=False, no_translate=False,
        voice='ru-RU-SvetlanaNeural', pause=0.7, max_len=15.0):
    out = Path(out)
    if ROOT in out.resolve().parents:
        sys.exit('видео с реальными людьми в репозиторий нельзя — выбери путь на D:')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='gesturelab_check_'))

    arr, fps, (w, h) = draw_source(video, tmp / 'skeleton.mp4', pause, max_len)
    spans = split_utterances(arr, fps, pause=pause, max_len=max_len)
    refs = [r.strip() for r in reference.split('|')] if reference else []
    rec = Recognizer(model, **{**TUNED_DECODER, 'beam': beam, 'nbest': nbest})
    tr = None if no_translate else Translator()

    preds, texts, rows = [], [], []
    for k, (a, b) in enumerate(spans):
        glosses, chains = rec.decode(arr[a:b])
        pred = [g for g, _ in glosses]
        text = ' '.join(pred)
        if tr:
            try:
                text = translate_in_context(tr, glosses, [], SYSTEM, chains=chains)
            except Exception as e:  # сеть/квота: показать глоссы, но не падать
                print(f'перевод недоступен: {e}', flush=True)
        preds.append(pred)
        texts.append(text)
        ref = refs[k] if k < len(refs) else ''
        n, recall, precision = hits(pred, ref) if ref else (0, 0.0, 0.0)
        row = {'span': [round(a / fps, 2), round(b / fps, 2)], 'reference': ref,
               'glosses': pred, 'translation': text, 'word_hits': n,
               'word_recall': round(recall, 3), 'word_precision': round(precision, 3),
               'chrf': round(chrf(text, ref), 3) if ref else 0.0}
        rows.append(row)
        print(f"{row['span'][0]}–{row['span'][1]} с: {pred} → {text}"
              + (f" | эталон {ref!r}: {n} слов" if ref else ''), flush=True)
    if refs and len(refs) != len(spans):
        print(f'ВНИМАНИЕ: эталонов {len(refs)}, реплик {len(spans)} — сопоставлены по порядку',
              flush=True)

    silent = tmp / 'skeleton.mp4'
    cap = cv2.VideoCapture(str(silent))
    final_silent = tmp / 'panel.mp4'
    writer = cv2.VideoWriter(str(final_silent), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (w, h + BAR))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(panel(frame, rows_for(i, spans, refs, preds, texts)))
        i += 1
    cap.release()
    writer.release()

    voices = []
    if not no_tts:
        for k, (a, _b) in enumerate(spans):
            if not texts[k] or texts[k] == '…':
                continue
            mp3 = tmp / f'v{k}.mp3'
            try:
                _tts(texts[k], mp3, voice)
                voices.append((mp3, a / fps))
            except Exception as e:  # озвучка не должна отменять результат
                print(f'озвучка недоступна: {e}', flush=True)
    mux(final_silent, voices, out, len(arr) / fps)

    result = {'video': Path(video).name, 'model': Path(model).name,
              'config': {'beam': beam, 'nbest': nbest, 'pause': pause, 'fps': round(fps, 2),
                         'decoder': {k: v for k, v in TUNED_DECODER.items()
                                     if k not in ('beam', 'nbest')},
                         'tts': not no_tts, 'translate': not no_translate},
              'frames': int(len(arr)), 'hands_ratio': round(edges(arr, fps), 3),
              'spans': rows, 'cost_usd': round(tr.cost, 4) if tr else 0.0, 'out': str(out)}
    stem = Path(video).stem
    Path('reports/evidence').mkdir(parents=True, exist_ok=True)
    Path(f'reports/evidence/check_{stem}.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'→ {out}', flush=True)
    return result


def selftest():
    img = np.zeros((60, 200, 3), np.uint8)
    out = panel(img, [(REF_COLOR, 'эталон: мама я тебя очень люблю'),
                      (GLOSS_COLOR, 'модель: мать я их'),
                      (TEXT_COLOR, 'перевод: Мама, у меня есть')])
    assert out.shape == (60 + BAR, 200, 3), out.shape
    assert out[60:].any(), 'полоса пустая'
    assert out[:60].any() == False  # кадр не тронут  # noqa: E712
    rows = rows_for(0, [], [], [], [])
    assert rows[0][1].startswith('ждём'), rows
    rows = rows_for(30, [(10, 50)], ['мама'], [['мать']], ['Мама.'])
    assert 'мама' in rows[0][1] and 'мать' in rows[1][1] and 'Мама' in rows[2][1], rows
    assert wrap('а б в г д', 4) == ['а б', 'в г…'], wrap('а б в г д', 4)
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', help='ролик РЖЯ, путь C:\\... или D:\\...')
    ap.add_argument('--reference', default='', help='эталон; реплики через |')
    ap.add_argument('--out', help='выходной mp4 (только вне репозитория)')
    ap.add_argument('--model', default=str(ROOT / 'models' / 'slovo_ctc_motion.pt'))
    ap.add_argument('--beam', type=int, default=24)
    ap.add_argument('--nbest', type=int, default=20)
    ap.add_argument('--pause', type=float, default=0.7)
    ap.add_argument('--max-utterance', type=float, default=15.0)
    ap.add_argument('--voice', default='ru-RU-SvetlanaNeural')
    ap.add_argument('--no-tts', action='store_true')
    ap.add_argument('--no-translate', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.video:
        sys.exit('нужен --video')
    out = a.out or str(Path('D:/hfdata/verify') / f'{Path(a.video).stem}_check.mp4')
    run(a.video, a.reference, out, a.model, a.beam, a.nbest, a.no_tts, a.no_translate,
        a.voice, a.pause, a.max_utterance)


if __name__ == '__main__':
    main()
