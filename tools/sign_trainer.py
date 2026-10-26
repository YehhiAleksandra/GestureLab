"""Тренажёр: эталонный жест → твой повтор → что увидела модель.

Для тех, кто не знает РЖЯ: слева крутится эталонный клип Slovo со скелетом — ровно те
59 точек, которые читает CTC-модель; справа твоя камера с тем же скелетом. Нажал Space,
повторил жест, получил вердикт: «совпало» / «почти» / «модель увидела другое». Так видно,
читает модель жест или выдаёт случайное слово, не умея языка.

Словарь тренажёра — только слова, эталонный клип которых production-модель распознаёт
с первого места после того же пути, что и камера (MediaPipe + beam-декодер). Иначе
непонятно, кто виноват: повтор или модель.

Клавиши: Space — записать повтор (повторное нажатие — остановить раньше), 0 — контроль
без жеста (руки в кадре, не двигаешь: модель должна молчать), →/↓ — следующее слово,
←/↑ — предыдущее (N/P тоже работают; в русской раскладке — т/з), R — проиграть эталон
заново, Q/Esc — выход.

  python tools/sign_trainer.py --camera 0
  python tools/sign_trainer.py --dry-run          # проверить первые 5 слов без камеры
  python tools/sign_trainer.py --rebuild-roster   # пересобрать словарь (минуты)
  python tools/sign_trainer.py --selftest

Видео Slovo и записи с камеры остаются на диске; в репозиторий идёт только JSON словаря
(слова и id клипов), без кадров.
"""
import argparse
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from gl_capture import fps_of, open_camera, open_video  # noqa: E402
from make_dialogue_demo import fit  # noqa: E402

REF_W, PANE_H = 338, 600          # эталон 9:16 вписывается без полей
CAM_W = 1280 - REF_W
BAR = 190
TARGET_COLOR = (120, 255, 120)    # BGR: зелёный — слово-задание
MODEL_COLOR = (0, 255, 255)       # жёлтый — что увидела модель
TALLY_COLOR = (190, 190, 190)
STATE_COLORS = {'watch': (230, 230, 230), 'record': (0, 255, 255),
                'match': (80, 230, 80), 'near': (0, 255, 255),
                'miss': (80, 80, 255), 'empty': (160, 160, 160)}

# Коды стрелок в cv2.waitKey на Windows (полный код, маску 0xFF не применять)
ARROW_LEFT, ARROW_RIGHT, ARROW_UP, ARROW_DOWN = 2424832, 2555904, 2490368, 2621440


def key_action(key):
    """Код waitKey → действие. В русской раскладке N/P приходят как т/з, поэтому
    есть ещё стрелки и Enter — они от раскладки не зависят."""
    if key in (ord('q'), ord('й'), 27):
        return 'quit'
    if key == 32:
        return 'record'
    if key == ord('0'):
        return 'control'
    if key in (ord('n'), ord('т'), ARROW_RIGHT, ARROW_DOWN, 13):
        return 'next'
    if key in (ord('p'), ord('з'), ARROW_LEFT, ARROW_UP):
        return 'prev'
    if key in (ord('r'), ord('к')):
        return 'replay'
    return None


def norm(word):
    """Сравнимое слово: регистр, ё→е, лишние пробелы."""
    return ' '.join(str(word).lower().replace('ё', 'е').split())


def verdict(target, glosses, chains=None):
    """(вид, текст) для одного повтора: совпало / почти / мимо / пусто."""
    top = [g for g, _ in glosses]
    t = norm(target)
    if not top:
        return 'empty', 'модель не увидела жеста'
    if norm(top[0]) == t:
        return 'match', f'СОВПАЛО: модель прочитала «{top[0]}»'
    near = any(norm(g) == t for g in top[:3]) or any(
        norm(g) == t for c in (chains or [])[:10] for g in c)
    if near:
        return 'near', f'ПОЧТИ: «{target}» была в вариантах, но выбрала «{top[0]}»'
    return 'miss', f'МИМО: модель увидела «{top[0]}»'


def present(arr):
    """Кадры, где видна хотя бы одна кисть."""
    t = len(arr)
    hands = np.asarray(arr)[:, :42].reshape(t, 2, 21, 3)
    return ~np.isnan(hands[:, :, 0, 0]).all(1)


def trim(arr, pad=0):
    """Срезать кадры без рук по краям — как нарезка реплики в production."""
    arr = np.asarray(arr)
    idx = np.where(present(arr))[0]
    if not len(idx):
        return arr[:0]
    a, b = max(0, idx[0] - pad), min(len(arr), idx[-1] + pad + 1)
    return arr[a:b]


def hands_ratio(arr):
    return float(present(arr).mean()) if len(arr) else 0.0


def load_reference(path, size=(REF_W, PANE_H)):
    """Эталонный mp4 → кадры со скелетом и ландмарки (тем же путём, что и камера)."""
    from dialogue_translator import frame_of
    from skeleton_overlay import draw_holistic

    cap = open_video(str(path))
    fps = fps_of(cap, 30.0)
    frames, arrs = [], []
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                              smooth_landmarks=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            arr = frame_of(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            small = fit(frame, *size)
            hands = arr[:42].reshape(2, 21, 3)
            draw_holistic(small, hands[0], hands[1], arr[42:49], arr[49:])
            frames.append(small)
            arrs.append(arr)
    finally:
        holistic.close()
        cap.release()
    if not frames:
        sys.exit(f'в клипе нет кадров: {path}')
    return {'frames': np.stack(frames), 'arr': np.stack(arrs), 'fps': fps}


def candidates(npz_path, limit):
    """Частотные слова корпуса, у которых есть тестовый клип Slovo. → ([(слово, кадр)], npz)."""
    from gloss_to_text import load_corpus

    z = np.load(str(npz_path), mmap_mode='r')
    texts, train = z['texts'], z['train']
    first = {}
    for i in np.where(~train)[0]:
        w = norm(texts[i])
        if w and w not in first:
            first[w] = int(i)
    freq = Counter()
    for p in load_corpus('train'):
        for w in p['rsl'].split():
            freq[norm(w)] += 1
    return [(w, first[w]) for w, _n in freq.most_common() if w in first][:limit], z


def build_roster(npz_path, videos, model, limit, rec):
    """Словарь: слово, id эталонного клипа, уверенность модели на этом клипе."""
    words = []
    t0 = time.time()
    cand, z = candidates(npz_path, limit)
    offsets, ids = z['offsets'], z['ids']
    for k, (word, i) in enumerate(cand, 1):
        # дешёвый предфильтр по разметке HF — не гоняем MediaPipe по всем словам
        glosses, _ = rec.decode(z['frames'][int(offsets[i]):int(offsets[i + 1])]
                                .astype(np.float32))
        if not glosses or norm(glosses[0][0]) != word:
            continue
        vid = str(ids[i])
        ref = load_reference(Path(videos) / f'{vid}.mp4')
        glosses, _ = rec.decode(ref['arr'])
        if glosses and norm(glosses[0][0]) == word:
            words.append({'word': word, 'id': vid, 'p': round(glosses[0][1], 3)})
        if k % 20 == 0:
            print(f'словарь: {k}/{len(cand)}, подходит {len(words)}, '
                  f'{time.time() - t0:.0f} с', flush=True)
    words.sort(key=lambda e: -e['p'])
    print(f'словарь готов: {len(words)} из {len(cand)} проверенных, '
          f'{time.time() - t0:.0f} с', flush=True)
    return {'built': date.today().isoformat(), 'model': Path(model).name,
            'source': 'datasets/slovo/test', 'limit': limit, 'tested': len(cand),
            'words': words}


def load_roster(path, videos, model, limit, rec, rebuild=False):
    p = Path(path)
    if p.exists() and not rebuild:
        roster = json.loads(p.read_text(encoding='utf-8'))
        if roster.get('words'):
            print(f"словарь: {len(roster['words'])} слов ({p.name}, {roster['built']})",
                  flush=True)
            return roster
    roster = build_roster(ROOT / 'datasets/slovo/slovo_holistic.npz', videos, model,
                          limit, rec)
    if not roster['words']:
        sys.exit('словарь пуст: ни одно слово не распознано на эталоне')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(roster, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'словарь записан: {p}', flush=True)
    return roster


def check_roster(roster, videos, rec, n):
    """Проверка без камеры: эталонные клипы должны читаться моделью с первого места."""
    bad = 0
    for e in roster['words'][:n]:
        ref = load_reference(Path(videos) / f"{e['id']}.mp4")
        glosses, chains = rec.decode(ref['arr'])
        kind, text = verdict(e['word'], glosses, chains)
        bad += kind != 'match'
        print(f"{e['word']}: {text} | руки {hands_ratio(ref['arr']):.0%} "
              f"| в словаре p={e['p']}", flush=True)
    print(f'проверено {min(n, len(roster["words"]))}, не совпало {bad}', flush=True)
    return bad


def caption(img, text, org, color=(255, 255, 255)):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1, cv2.LINE_AA)


def rows_for(word, p, kind, state_text, glosses, stats):
    """Строки полосы: задание, состояние, что увидела модель, счёт сессии."""
    model = ' '.join(f'{g} ({c:.2f})' for g, c in glosses[:2]) or '—'
    tally = (f"повтор {stats['match']}/{stats['attempts']} совпал"
             f" · почти {stats['near']} · мимо {stats['miss']}"
             f" · контроль {stats['controls']}, галлюцинаций {stats['halluc']}")
    return [(TARGET_COLOR, f'слово: {word} · эталон модель читает ✓ {p:.2f}'),
            (STATE_COLORS.get(kind, STATE_COLORS['watch']), state_text),
            (MODEL_COLOR, f'модель: {model}'),
            (TALLY_COLOR, tally)]


def set_reference(state, entries, k, videos):
    """Переключить слово: загрузить эталон и вернуть (кадры, момент старта цикла)."""
    ref = load_reference(Path(videos) / f"{entries[k]['id']}.mp4")
    state['kind'], state['glosses'] = 'watch', []
    state['text'] = 'смотри эталон · Space — повторить · → — дальше · 0 — контроль'
    print(f"[{k + 1}/{len(entries)}] {entries[k]['word']} (p={entries[k]['p']})", flush=True)
    return ref, time.time()


def attempt_result(entry, attempt, fps, rec, stats):
    """Записанный повтор → (вид, текст, глоссы); счёт сессии обновляется здесь."""
    frames = trim(np.stack(attempt['buf'])) if attempt['buf'] else None
    if frames is None or len(frames) < max(6, round(0.2 * fps)):
        return 'empty', 'руки не попали в кадр — попытка не засчитана', []
    glosses, chains = rec.decode(frames)
    if attempt['mode'] == 'control':
        if glosses:
            stats['halluc'] += 1
            text = (f'КОНТРОЛЬ: модель «увидела» '
                    f'{" ".join(g for g, _ in glosses)} — это галлюцинация')
            print(f'контроль: {text}', flush=True)
            return 'miss', text, glosses
        stats['controls'] += 1
        text = 'КОНТРОЛЬ: модель молчит — правильно'
        print(f'контроль: {text}', flush=True)
        return 'match', text, glosses
    kind, text = verdict(entry['word'], glosses, chains)
    if kind == 'empty':
        return 'empty', 'модель не увидела жеста — попытка не засчитана', []
    stats['attempts'] += 1
    stats[kind] += 1
    print(f"повтор «{entry['word']}»: {text}", flush=True)
    return kind, text, glosses


def live(camera, rec, roster, videos, seconds=4.0):
    from check_translation import panel
    from dialogue_translator import frame_of
    from skeleton_overlay import draw_holistic

    entries = roster['words']
    cap = open_camera(camera)
    fps = fps_of(cap)
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                              smooth_landmarks=True)
    stats = {'attempts': 0, 'match': 0, 'near': 0, 'miss': 0,
             'controls': 0, 'halluc': 0}
    idx = 0
    st = {'attempt': None, 'kind': 'watch', 'glosses': [], 'text': ''}
    seen_hist = []
    ref, ref_t0 = set_reference(st, entries, 0, videos)
    print('Space — повтор, 0 — контроль без жеста, →/↓ — дальше, ←/↑ — назад, '
          'R — эталон, Q — выход', flush=True)
    print('Если клавиши не доходят — кликни по окну тренажёра (фокус уходит в консоль)',
          flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            arr = frame_of(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cam = fit(frame, CAM_W, PANE_H)
            hands = arr[:42].reshape(2, 21, 3)
            seen = not np.isnan(hands[:, 0, 0]).all()
            seen_hist = (seen_hist + [seen])[-90:]
            draw_holistic(cam, hands[0], hands[1], arr[42:49], arr[49:])

            if st['attempt'] is not None:
                st['attempt']['buf'].append(arr)
                elapsed = time.time() - st['attempt']['started']
                st['kind'] = 'record'
                st['text'] = (f'запись {elapsed:.1f} / {seconds:.1f} с — показывай жест '
                              f'(Space — остановить)')
                if elapsed >= seconds:
                    st['kind'], st['text'], st['glosses'] = attempt_result(
                        entries[idx], st['attempt'], fps, rec, stats)
                    st['attempt'] = None

            entry = entries[idx]
            ref_img = ref['frames'][int((time.time() - ref_t0) * ref['fps'])
                                    % len(ref['frames'])]
            comp = np.hstack([ref_img, cam])
            caption(comp, 'ЭТАЛОН', (10, 30))
            caption(comp, 'ТЫ', (REF_W + 10, 30))
            vis = float(np.mean(seen_hist))
            caption(comp, f'руки в кадре {vis:.0%}', (REF_W + 10, 58),
                    (120, 220, 120) if vis > 0.5 else (80, 80, 255))
            out = panel(comp, rows_for(entry['word'], entry['p'], st['kind'], st['text'],
                                       st['glosses'], stats), height=BAR)
            cv2.imshow('GestureLab sign trainer', out)
            act = key_action(cv2.waitKey(1))
            if act == 'quit':
                break
            if act == 'record':
                if st['attempt'] is None:
                    st['attempt'] = {'mode': 'repeat', 'buf': [], 'started': time.time()}
                    st['glosses'] = []
                elif len(st['attempt']['buf']) >= 0.3 * fps:
                    st['kind'], st['text'], st['glosses'] = attempt_result(
                        entries[idx], st['attempt'], fps, rec, stats)
                    st['attempt'] = None
            elif act == 'control' and st['attempt'] is None:
                st['attempt'] = {'mode': 'control', 'buf': [], 'started': time.time()}
                st['glosses'] = []
            elif act == 'next' and st['attempt'] is None:
                idx = (idx + 1) % len(entries)
                ref, ref_t0 = set_reference(st, entries, idx, videos)
            elif act == 'prev' and st['attempt'] is None:
                idx = (idx - 1) % len(entries)
                ref, ref_t0 = set_reference(st, entries, idx, videos)
            elif act == 'replay':
                ref_t0 = time.time()
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()
        print(f"сессия: повтор {stats['match']}/{stats['attempts']} совпал, "
              f"почти {stats['near']}, мимо {stats['miss']}, "
              f"контроль {stats['controls']} / галлюцинаций {stats['halluc']}", flush=True)


def selftest():
    assert norm('  ВОДА ') == 'вода' and norm('Ёлка') == 'елка'
    assert verdict('вода', [('вода', 0.9)])[0] == 'match'
    assert verdict('вода', [('еда', 0.9)], [['еда'], ['вода']])[0] == 'near'
    assert verdict('вода', [('еда', 0.9)])[0] == 'miss'
    assert verdict('вода', [])[0] == 'empty'

    arr = np.full((10, 59, 3), np.nan, np.float32)
    arr[3:7, 0, 0] = 0.5
    assert len(trim(arr)) == 4 and hands_ratio(trim(arr)) == 1.0
    assert len(trim(np.full((5, 59, 3), np.nan, np.float32))) == 0

    assert fit(np.zeros((100, 200, 3), np.uint8), 100, 100).shape == (100, 100, 3)

    stats = {'attempts': 2, 'match': 1, 'near': 0, 'miss': 1, 'controls': 1, 'halluc': 0}
    rows = rows_for('вода', 0.9, 'match', 'СОВПАЛО', [('вода', 0.9)], stats)
    assert len(rows) == 4 and 'вода' in rows[0][1] and 'совпал' in rows[3][1], rows
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int)
    ap.add_argument('--model', default=str(ROOT / 'models' / 'slovo_ctc_motion.pt'))
    ap.add_argument('--videos', default=str(ROOT / 'datasets/slovo/test'))
    ap.add_argument('--roster', default=str(ROOT / 'reports/evidence/trainer_roster.json'))
    ap.add_argument('--limit', type=int, default=200,
                    help='сколько частотных слов корпуса проверять при сборке словаря')
    ap.add_argument('--seconds', type=float, default=4.0, help='длина записи повтора, с')
    ap.add_argument('--rebuild-roster', action='store_true')
    ap.add_argument('--dry-run', type=int, nargs='?', const=5, metavar='N',
                    help='проверить первые N эталонов без камеры')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    from dialogue_translator import Recognizer  # torch грузим только после selftest
    rec = Recognizer(a.model, decoder='beam', beam=24, lm_weight=0.1, word_bonus=2.0,
                     smooth=1, nbest=10)
    roster = load_roster(a.roster, a.videos, a.model, a.limit, rec, a.rebuild_roster)
    if a.dry_run is not None:
        sys.exit(1 if check_roster(roster, a.videos, rec, a.dry_run) else 0)
    if a.camera is None:
        sys.exit('нужен --camera 0 (или --dry-run для проверки без камеры)')
    live(a.camera, rec, roster, a.videos, a.seconds)


if __name__ == '__main__':
    main()
