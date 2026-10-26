"""Инвентаризация Rutube-канала 23691449 и OCR вшитых подписей.

Канал: 894 словарных ролика (слово в заголовке), 67 фразовых клипов по ~3 с,
33 буквы алфавита и 7 длинных информационных/музыкальных роликов. Tesseract читает
вшитые подписи по кадрам и отдаёт отрезки «текст + таймкод». На этом канале подписи
длинных роликов — имена интервьюируемых, текст песни и титры заставки, а не глоссы:
пригодность каждого ролика проверяется глазами (см. reports/19_stage13_channel_ocr.md).
Видео и кадры остаются на D:, в репозиторий идут только текст и таймкоды.

  python tools/rt_channel.py --inventory
  python tools/rt_channel.py --ocr-dir D:\\hfdata\\rutube_phrases
  python tools/rt_channel.py --ocr-video D:\\hfdata\\demo_frases\\vazhno.mp4
  python tools/rt_channel.py --selftest
"""
import argparse
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import pytesseract

CHANNEL = 'https://rutube.ru/channel/23691449/'
CACHE = Path('D:/hfdata/rutube_titles.txt')
PREFIX = {'dictionary': 'ЖЕСТОВЫЙ ЯЗЫК. СЛОВАРЬ.',
          'phrase': 'ЖЕСТОВЫЙ ЯЗЫК. ФРАЗЫ.',
          'alphabet': 'ЖЕСТОВЫЙ ЯЗЫК. АЛФАВИТ.'}
LONG_TITLES = {'О важном на жестовом языке', 'Без слов о самом главном',
               'Жестовый язык. Фразы. Мама я тебя очень люблю!!!',
               'Поздравление для мамы на языке жестов!',
               'Изучение жестового и иностранных языков в Академии МВД',
               'Основы жестового языка преподают в Академии МВД. Архивное видео',
               '22вагон.mpg'}


def kind_of(title: str) -> str:
    for kind, prefix in PREFIX.items():
        if title.startswith(prefix):
            return kind
    return 'long' if title in LONG_TITLES else 'other'


def label_of(title: str) -> str:
    for prefix in PREFIX.values():
        if title.startswith(prefix):
            return title[len(prefix):].strip(' .')
    return title


def parse_titles(path=CACHE) -> list:
    out = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if '|' not in line:
            continue
        vid, title = line.strip().split('|', 1)
        out.append({'id': vid, 'title': title, 'kind': kind_of(title),
                    'label': label_of(title)})
    return out


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return {'fps': round(fps, 2), 'duration': round(n / fps, 2), 'size': list(size)}


def inventory(media_dir, out):
    items = parse_titles()
    by_kind = {k: [i for i in items if i['kind'] == k] for k in
               ('dictionary', 'phrase', 'alphabet', 'long', 'other')}
    media = Path(media_dir)
    downloaded = []
    if media.exists():
        for p in sorted(media.glob('*.mp4')):
            meta = video_meta(p)
            downloaded.append({'id': p.stem, **meta})
    dur = {d['id']: d['duration'] for d in downloaded}
    summary = {
        'channel': CHANNEL,
        'date': time.strftime('%Y-%m-%d'),
        'counts': {k: len(v) for k, v in by_kind.items()},
        'duration_downloaded_s': round(sum(dur.values()), 1),
        'phrases': [{'id': i['id'], 'phrase': i['label'],
                     'duration': dur.get(i['id'])} for i in by_kind['phrase']],
        'long': [{'id': i['id'], 'title': i['title'],
                  'duration': dur.get(i['id'])} for i in by_kind['long']],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('counts', 'duration_downloaded_s')},
                     ensure_ascii=False), flush=True)
    for i in by_kind['long']:
        print(f"  long  {dur.get(i['id'], '—'):>7} с  {i['title']}", flush=True)
    print(f'→ {out}', flush=True)
    return summary


def clean(text: str) -> str:
    text = re.sub(r'[^А-Яа-яЁёA-Za-z0-9 ,.!?;:—-]', ' ', text)
    text = re.sub(r'(?<=[А-Яа-яЁё])(?=\d)|(?<=\d)(?=[А-Яа-яЁё])', ' ', text)
    return re.sub(r'\s+', ' ', text).strip(' ,;:-')


def _words(img, min_conf):
    d = pytesseract.image_to_data(img, lang='rus', config='--psm 6',
                                  output_type=pytesseract.Output.DICT)
    out = []
    for w, c in zip(d['text'], d['conf']):
        try:
            c = int(float(c))
        except ValueError:
            continue
        if w.strip() and c >= min_conf:
            out.append((w.strip(), c))
    return out


def read_caption(frame, crop=0.72, min_conf=50):
    """Кадр → (текст подписи, сумма уверенностей). Пусто, если подписи нет.

    Две бинаризации: белый текст (информационные ролики) и адаптивный порог,
    который читает цветные/тёмные титры (песни). Выбирается вариант с большей
    суммой уверенностей по словам с кириллицей. Белый порог — без размытия:
    blur превращает «более» в «оолее».
    """
    h = frame.shape[0]
    g = cv2.cvtColor(frame[int(h * crop):], cv2.COLOR_BGR2GRAY)
    variants = [
        255 - ((g > 200) * 255).astype(np.uint8),
        cv2.adaptiveThreshold(cv2.GaussianBlur(g, (3, 3), 0), 255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 10),
    ]
    best, best_score = '', 0
    for img in variants:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        words = _words(img, min_conf)
        text = clean(' '.join(w for w, _ in words))
        score = sum(c for w, c in words if re.search('[А-Яа-яЁё]', w))
        if score > best_score:
            best, best_score = text, score
    if len(re.findall('[А-Яа-яЁё]', best)) < 3:
        return '', 0
    return best, best_score


def merge(rows, step, same=0.8, gap=0.75, min_len=0.75):
    """[(время, текст)] → [(начало, конец, текст)] с той же подписью подряд.

    Внутри отрезка копятся варианты прочтения, побеждает самый частый: одиночный
    сбой OCR («вместо» вместо «вместе») не попадает в разметку.
    """
    segs = []
    for t, text in rows:
        if not text:
            continue
        if segs and t - segs[-1][1] <= gap and \
                SequenceMatcher(None, max(segs[-1][2], key=segs[-1][2].get), text).ratio() >= same:
            a, _, votes = segs[-1]
            votes[text] = votes.get(text, 0) + 1
            segs[-1] = (a, t, votes)
        else:
            segs.append((t, t, {text: 1}))
    out = []
    for a, b, votes in segs:
        if b + step - a >= min_len:
            out.append((a, b + step, max(votes, key=lambda x: (votes[x], len(x)))))
    return out


def ocr_video(path, fps=2.0, crop=0.72, min_conf=50):
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(vfps / fps))
    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            text, _ = read_caption(frame, crop, min_conf)
            rows.append((i / vfps, text))
        i += 1
    cap.release()
    return merge(rows, step / vfps)


def srt(segments):
    def ts(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f'{int(h):02d}:{int(m):02d}:{s:06.3f}'.replace('.', ',')
    return '\n\n'.join(f'{i + 1}\n{ts(a)} --> {ts(b)}\n{tx}'
                       for i, (a, b, tx) in enumerate(segments)) + '\n'


def save(path, segments):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(
        [{'start': round(a, 2), 'end': round(b, 2), 'text': tx} for a, b, tx in segments],
        ensure_ascii=False, indent=1), encoding='utf-8')
    out.with_suffix('.srt').write_text(srt(segments), encoding='utf-8')


def ocr_many(media_dir, out_dir, fps=2.0, crop=0.72, min_conf=50,
             min_duration=15.0, force=False):
    media, out = Path(media_dir), Path(out_dir)
    files = sorted(media.glob('*.mp4'))
    done = total = 0
    started = time.time()
    for p in files:
        meta = video_meta(p)
        if meta['duration'] < min_duration:
            continue
        target = out / p.stem
        if target.with_suffix('.json').exists() and not force:
            continue
        total += 1
        segs = ocr_video(p, fps, crop, min_conf)
        save(target, segs)
        done += len(segs)
        chars = sum(len(tx) for _, _, tx in segs)
        print(f'{p.stem} {meta["duration"]:.0f} с: {len(segs)} подписей, {chars} симв. '
              f'({time.time() - started:.0f} с)', flush=True)
    print(f'итого {total} роликов, {done} подписей', flush=True)


def selftest():
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(img, 'ТЕСТ ПОДПИСИ', (60, 430), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (255, 255, 255), 3, cv2.LINE_AA)
    text, score = read_caption(img)
    assert 'ТЕСТ' in text, f'OCR не прочитал синтетическую подпись: {text!r}'
    assert merge([(0.0, 'А'), (0.5, 'А'), (1.0, 'Б'), (1.5, 'Б')], 0.5) == \
        [(0.0, 1.0, 'А'), (1.0, 2.0, 'Б')]
    print('selftest ok', repr(text), score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inventory', action='store_true')
    ap.add_argument('--media', default='D:/hfdata/rutube_phrases')
    ap.add_argument('--inventory-out',
                    default='reports/evidence/rutube_channel_inventory.json')
    ap.add_argument('--ocr-dir')
    ap.add_argument('--ocr-video')
    ap.add_argument('--out', default='D:/hfdata/rutube_phrases/ocr')
    ap.add_argument('--fps', type=float, default=2.0)
    ap.add_argument('--crop', type=float, default=0.72)
    ap.add_argument('--min-conf', type=int, default=50)
    ap.add_argument('--min-duration', type=float, default=15.0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.inventory:
        inventory(a.media, a.inventory_out)
    elif a.ocr_video:
        segs = ocr_video(a.ocr_video, a.fps, a.crop, a.min_conf)
        stem = Path(a.ocr_video).stem
        save(Path(a.out) / stem, segs)
        for s in segs:
            print(f'{s[0]:6.1f}-{s[1]:6.1f} {s[2]}', flush=True)
    elif a.ocr_dir:
        ocr_many(a.ocr_dir, a.out, a.fps, a.crop, a.min_conf,
                 a.min_duration, a.force)
    else:
        sys.exit('нужен --inventory, --ocr-dir, --ocr-video или --selftest')


if __name__ == '__main__':
    main()
