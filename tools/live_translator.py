"""Живой переводчик РЖЯ → русская речь: камера → жесты → фраза (LLM) → озвучка.

Как показывать: жест в кадре, между жестами опустить руки (или нажать Пробел).
Клавиши: 1/2/3 — выбрать вариант для последнего жеста, Backspace — удалить его,
Enter — перевести и озвучить фразу, C — очистить, Q/Esc — выход.
--video файл — тот же конвейер на записанном видео без окна (проверка без камеры).
Наружу уходят только глоссы (HF router) и готовая фраза (edge-tts), не видео и не точки.
"""
import argparse
import asyncio
import ctypes
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from live_signs import SignSegmenter, hands_array  # noqa: E402
from slovo_model import SignNet, features, resample  # noqa: E402


class Recognizer:
    def __init__(self, path, k=3):
        ck = torch.load(path)
        self.names, self.k = ck['names'], k
        self.model = SignNet(len(self.names))
        self.model.load_state_dict(ck['model'])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, seg):
        p = self.model(torch.from_numpy(features(resample(seg)))[None]).softmax(1)[0]
        v, i = p.topk(self.k)
        return [(self.names[j], float(s)) for j, s in zip(i.tolist(), v.tolist())]


def speak(text, voice='ru-RU-SvetlanaNeural'):
    import edge_tts
    path = Path(tempfile.gettempdir()) / 'gesturelab_tts.mp3'
    asyncio.run(edge_tts.Communicate(text, voice).save(str(path)))
    if sys.platform == 'win32':  # MCI играет mp3 штатно, без новых зависимостей
        mci = ctypes.windll.winmm.mciSendStringW
        mci(f'open "{path}" type mpegvideo alias gl_tts', None, 0, None)
        mci('play gl_tts wait', None, 0, None)
        mci('close gl_tts', None, 0, None)


class Phrase:
    def __init__(self, translator, tts):
        self.signs, self.text, self.busy = [], '', False
        self.translator, self.tts = translator, tts

    def add(self, cands):
        self.signs.append({'cands': cands, 'pick': None})

    def glosses(self):
        return [s['cands'][s['pick'] or 0][0] for s in self.signs]

    def translate(self, wait=False):
        if not self.signs or self.busy:
            return
        lattice = [[s['cands'][s['pick']][0]] if s['pick'] is not None else [c[0] for c in s['cands']]
                   for s in self.signs]
        self.busy, self.text = True, '…'

        def run():
            try:
                self.text = self.translator().lattice(lattice)
                print('ФРАЗА:', self.text, flush=True)
                if self.tts:
                    speak(self.text)
            except Exception as e:  # сеть/квота: показать, не падать
                self.text = f'ошибка перевода: {e}'
            finally:
                self.busy = False
        t = threading.Thread(target=run, daemon=True)
        t.start()
        if wait:
            t.join()


def draw(frame, phrase, recording):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 110), (w, h), (0, 0, 0), -1)
    if recording:
        cv2.circle(frame, (20, 20), 9, (0, 0, 255), -1)
    if phrase.signs:
        last = phrase.signs[-1]
        alts = '  '.join(f'{i + 1}:{n} {p:.0%}' for i, (n, p) in enumerate(last['cands']))
        cv2.putText(frame, alts, (10, h - 80), cv2.FONT_HERSHEY_COMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, ' '.join(phrase.glosses()), (10, h - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 1)
    cv2.putText(frame, phrase.text, (10, h - 15), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--video', help='видеофайл вместо камеры (без окна), путь в формате C:\\...')
    ap.add_argument('--model', default=str(ROOT / 'models' / 'slovo_signnet.pt'))
    ap.add_argument('--no-tts', action='store_true')
    ap.add_argument('--gap', type=int, default=6, help='кадров без рук = конец жеста')
    a = ap.parse_args()

    rec = Recognizer(a.model)
    holder = {}

    def translator():  # корпус few-shot качается с HF один раз и только когда нужен перевод
        if 'tr' not in holder:
            from gloss_to_text import Translator
            holder['tr'] = Translator()
        return holder['tr']

    phrase = Phrase(translator, tts=not a.no_tts)
    seg = SignSegmenter(gap=a.gap)
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        sys.exit(f'не открылся источник: {a.video or a.camera}')
    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)

    def on_sign(s):
        if s is not None:
            cands = rec(s)
            phrase.add(cands)
            print('жест:', ', '.join(f'{n} {p:.0%}' for n, p in cands), flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        on_sign(seg.push(hands_array(res.multi_hand_landmarks)))
        if a.video:
            continue
        for hl in res.multi_hand_landmarks or []:
            mp.solutions.drawing_utils.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
        draw(frame, phrase, bool(seg.buf))
        cv2.imshow('GestureLab RSL -> speech', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == 32:
            on_sign(seg.flush())
        elif key in (ord('1'), ord('2'), ord('3')) and phrase.signs:
            i = key - ord('1')
            if i < len(phrase.signs[-1]['cands']):
                phrase.signs[-1]['pick'] = i
        elif key == 8 and phrase.signs:
            phrase.signs.pop()
        elif key == 13:
            phrase.translate()
        elif key in (ord('c'), ord('C')):
            phrase.signs, phrase.text = [], ''
    on_sign(seg.flush())
    cap.release()
    cv2.destroyAllWindows()
    if a.video:
        print('глоссы:', ' '.join(phrase.glosses()))
        phrase.translate(wait=True)


if __name__ == '__main__':
    sys.exit(main())
