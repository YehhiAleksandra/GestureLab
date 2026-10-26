"""Диалоговый переводчик РЖЯ: непрерывный поток жестов → реплики → контекстный перевод → голос.

Отличие от live_translator.py: жесты не нужно разделять опусканием рук — CTC-модель
(sign_stream) сама расставляет границы внутри непрерывного показа. Реплика заканчивается
паузой без рук; следующий перевод видит предыдущие реплики диалога, поэтому местоимения и
пропущенные слова восстанавливаются по контексту.

Режимы:
  --video файл       офлайн-разбор записи (по умолчанию без окна)
  --camera 0         живой режим с камерой
  --mic              слушать слышащую сторону микрофоном (faster-whisper), иначе клавиша H
  --hearing "..."    реплика слышащего до начала записи (контекст для первого перевода)
  --decoder beam     beam search с LM глоссов вместо greedy (--lm-weight — сила LM,
                     --word-bonus — бонус за слово, иначе LM режет короткие реплики)
  --variants chains  что видит переводчик из n-best: целые цепочки (по умолчанию), варианты
                     на каждое место (lattice) или ничего (none)
  --no-tts           не озвучивать перевод
Клавиши: Q/Esc — выход, H — ввести реплику слышащего, S — повторить озвучку, C — очистить.

Наружу уходят только глоссы и готовая фраза (HF router, edge-tts); видео и точки остаются
на машине. Реплики сохраняются в JSON, если указан --dialogue-out.
"""
import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from dialogue_state import Dialogue, translate_in_context  # noqa: E402
from gl_capture import open_camera, open_video  # noqa: E402
from live_signs import FACE_IDX, POSE_IDX, holistic_frame  # noqa: E402
from sign_beam import GlossLM, ctc_beam_search, nbest_to_lattice, smooth_log_probs  # noqa: E402
from sign_stream import greedy_decode, load_ctc  # noqa: E402
from sign_utils import FEAT, features, split_utterances  # noqa: E402


def frame_of(res):
    """Результат mp Holistic → [NPTS, 3]; кадр без рук тоже нужен — по нему видна пауза."""
    def pts(lm, idx=slice(None)):
        return None if lm is None else np.array(
            [(p.x, p.y, p.z) for p in lm.landmark], np.float32)[idx]
    out = holistic_frame(pts(res.left_hand_landmarks), pts(res.right_hand_landmarks),
                         pts(res.pose_landmarks, POSE_IDX), pts(res.face_landmarks, FACE_IDX))
    return out


class Recognizer:
    def __init__(self, path, device=None, decoder='greedy', beam=8, lm_weight=0.0,
                 word_bonus=0.0, smooth=1, nbest=1):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.names, ck = load_ctc(path, self.device)
        self.feat = ck.get('feat', FEAT)
        self.decoder, self.beam = decoder, beam
        self.lm_weight, self.word_bonus = lm_weight, word_bonus
        self.smooth, self.lm = smooth, None
        self.nbest = nbest if decoder == 'beam' else 1
        if decoder == 'beam' and lm_weight > 0:
            try:
                from gloss_to_text import load_corpus
                index = {n.lower(): i for i, n in enumerate(self.names)}
                seqs = [[index[w.lower()] for w in p['rsl'].split()] for p in load_corpus('train')
                        if all(w.lower() in index for w in p['rsl'].split())]
                self.lm = GlossLM(seqs, len(self.names))
                print(f'LM глоссов: {len(seqs)} фраз корпуса', flush=True)
            except Exception as e:  # noqa: BLE001 — без корпуса beam работает без LM
                print(f'LM недоступна ({e}) — beam без языковой модели', flush=True)
        print(f'CTC-модель: {Path(path).name}, классов {len(self.names)}, {self.device}, '
              f'декодер {decoder}', flush=True)

    @torch.no_grad()
    def decode(self, frames):
        """[T, NPTS, 3] → ([(глосса, уверенность)], варианты n-best или None)."""
        if not len(frames):
            return [], None
        x = torch.from_numpy(features(frames, self.feat))[None].to(self.device)
        lengths = torch.tensor([len(frames)], device=self.device)
        logits = self.model(x, lengths)[0].cpu()
        if self.decoder != 'beam':
            ids, scores = greedy_decode(logits)
            return [(self.names[i], c) for i, c in zip(ids, scores)], None
        logp = logits.log_softmax(-1).numpy()
        if self.smooth > 1:
            logp = smooth_log_probs(logp, self.smooth)
        hyps = ctc_beam_search(logp, beam=self.beam, topk=32, nbest=self.nbest,
                              lm=self.lm, lm_weight=self.lm_weight,
                              word_bonus=self.word_bonus)
        ids = hyps[0][0]
        p = np.exp(logp)
        scores = [float(p[:, i + 1].max()) for i in ids]  # верхняя оценка, не калибровка
        chains = None
        if self.nbest > 1:
            chains = [[self.names[i] for i in h] for h, _ in hyps]
        return [(self.names[i], c) for i, c in zip(ids, scores)], chains


class Worker:
    """Перевод и озвучка в фоне, чтобы камера не замирала на время ответа LLM."""

    def __init__(self, dialogue, tts=True, translator=True, variants='chains'):
        self.dialogue, self.tts, self.use_translator = dialogue, tts, translator
        self.variants = variants
        self.q, self.tr, self.system, self.stop = queue.Queue(), None, None, False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, glosses, at, candidates=None):
        self.q.put((glosses, at, candidates))

    def run(self):
        while not self.stop:
            try:
                glosses, at, candidates = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                text = ' '.join(g for g, _ in glosses) or '…'
                try:
                    if self.use_translator:
                        if self.tr is None:
                            from gloss_to_text import LATTICE, NBEST, SYSTEM, Translator
                            self.tr, self.system = Translator(), SYSTEM
                            self.prompts = {'chains': NBEST, 'lattice': LATTICE}
                        variant = self.variants if candidates else None
                        system = self.system + (self.prompts[variant] if variant else '')
                        text = translate_in_context(
                            self.tr, glosses, self.dialogue.context(), system,
                            chains=candidates if variant == 'chains' else None,
                            candidates=(nbest_to_lattice(candidates)
                                        if variant == 'lattice' else None))
                except Exception as e:  # сеть/квота: оставить глоссы, но не портить контекст
                    print(f'перевод недоступен: {e}', flush=True)
                self.dialogue.add('deaf', text, [g for g, _ in glosses],
                                  [round(c, 3) for _, c in glosses], at)
                print('ГЛУХОЙ:', ' '.join(g for g, _ in glosses), '→', text, flush=True)
                if self.tts:
                    try:
                        from live_translator import speak
                        speak(text)
                    except Exception as e:
                        print(f'озвучка недоступна: {e}', flush=True)
            finally:
                self.q.task_done()


class Mic:
    """Слышащая сторона: VAD по энергии → faster-whisper. Ошибки не валят переводчика."""

    BLOCK = 800  # 50 мс при 16 кГц

    def __init__(self, dialogue, model='small'):
        self.dialogue, self.model_name = dialogue, model
        self.last = np.zeros(self.BLOCK, np.float32)
        self.rec, self.recording = [], False
        self.lock = threading.Lock()
        self.model = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        try:
            import sounddevice as sd
        except Exception as e:
            print(f'микрофон недоступен: {e}', flush=True)
            return False
        self.sd = sd
        self.thread.start()
        print('микрофон слушает слышащую сторону', flush=True)
        return True

    def run(self):
        sd = self.sd

        def cb(indata, frames, t, status):
            x = indata.copy().reshape(-1)
            with self.lock:
                self.last = x
                if self.recording:
                    self.rec.append(x)

        with sd.InputStream(samplerate=16000, channels=1, dtype='float32',
                            blocksize=self.BLOCK, callback=cb):
            speaking, silence = False, 0
            while True:
                time.sleep(0.05)
                with self.lock:
                    energy = float(np.sqrt(np.mean(self.last ** 2)))
                if energy > 0.012:
                    if not speaking:
                        speaking = True
                        with self.lock:
                            self.recording, self.rec = True, [self.last]
                    silence = 0
                elif speaking:
                    silence += 1
                    if silence > 12:  # ~0.6 с тишины
                        with self.lock:
                            audio = np.concatenate(self.rec)
                            self.rec, self.recording = [], False
                        speaking, silence = False, 0
                        self.transcribe(audio)

    def transcribe(self, audio):
        try:
            if self.model is None:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.model_name, device='cpu', compute_type='int8')
            segs, _ = self.model.transcribe(audio, language='ru', vad_filter=True)
            text = ' '.join(s.text.strip() for s in segs).strip()
            if text:
                self.dialogue.add('hearing', text)
                print('СЛЫШАЩИЙ:', text, flush=True)
        except Exception as e:
            print(f'распознавание речи недоступно: {e}', flush=True)


def process_video(path, rec, dialogue, worker, fps=25.0, pause=0.7, max_len=15.0, show=False):
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frames, t0 = [], time.time()
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                              smooth_landmarks=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_of(holistic.process(rgb)))
        if show:
            cv2.imshow('GestureLab dialogue', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    cap.release()
    holistic.close()
    print(f'кадров {len(frames)}, извлечение {time.time() - t0:.0f} с', flush=True)
    if not frames:
        sys.exit(f'в видео нет кадров: {path}')

    arr = np.stack(frames)
    spans = split_utterances(arr, fps, pause=pause, max_len=max_len)
    print(f'реплик: {len(spans)}', flush=True)
    for a, b in spans:
        glosses, cands = rec.decode(arr[a:b])
        if glosses:
            print(f'{a / fps:.1f}–{b / fps:.1f} с:', ' '.join(g for g, _ in glosses), flush=True)
            worker.submit(glosses, f'{a / fps:.1f}–{b / fps:.1f} с', cands)
    worker.q.join()
    if show:
        cv2.destroyAllWindows()
    return dialogue


def live(camera, rec, dialogue, worker, pause=0.7, max_len=15.0):
    cap = open_camera(camera)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                              smooth_landmarks=True)
    buf, silent, state = [], 0, {'glosses': [], 'text': '…'}

    def flush():
        nonlocal buf
        if len(buf) < 0.6 * fps:
            buf = []
            return
        glosses, cands = rec.decode(np.stack(buf))
        buf = []
        if glosses:
            state['glosses'] = [g for g, _ in glosses]
            worker.submit(glosses, datetime.now().strftime('%H:%M:%S'), cands)

    print('Q/Esc — выход, H — реплика слышащего, S — повторить, C — очистить', flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        arr = frame_of(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        buf.append(arr)
        hands = arr[:42].reshape(2, 21, 3)[:, 0, 0]
        if np.isnan(hands).all():
            silent += 1
            if silent >= pause * fps:
                flush()
        else:
            silent = 0
        if len(buf) > max_len * fps:
            flush()
        hud(frame, state, buf, fps, dialogue)
        cv2.imshow('GestureLab dialogue', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('h'):
            text = input('Слышащий: ').strip()
            if text:
                dialogue.add('hearing', text)
        if key == ord('s') and state['text'] != '…':
            try:
                from live_translator import speak
                speak(state['text'])
            except Exception as e:
                print(f'озвучка недоступна: {e}', flush=True)
        if key == ord('c'):
            state['glosses'], state['text'] = [], '…'
    flush()
    worker.q.join()
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


def hud(frame, state, buf, fps, dialogue):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 120), (w, h), (0, 0, 0), -1)
    last = dialogue.turns[-1] if dialogue.turns else None
    if last and last['speaker'] == 'deaf':
        state['text'] = last['text']
    cv2.putText(frame, f'{len(buf) / fps:.1f} с в буфере', (10, 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, ' '.join(state['glosses'])[:90], (10, h - 80),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 1)
    cv2.putText(frame, state['text'][:90], (10, h - 45),
                cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 1)
    if last and last['speaker'] == 'hearing':
        cv2.putText(frame, 'Слышащий: ' + last['text'][:80], (10, h - 12),
                    cv2.FONT_HERSHEY_COMPLEX, 0.55, (180, 220, 180), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int)
    ap.add_argument('--video', help='видеофайл вместо камеры; путь в формате C:\\...')
    ap.add_argument('--model', default=str(ROOT / 'models' / 'slovo_ctc_motion.pt'))
    ap.add_argument('--pause', type=float, default=0.7, help='секунд без рук = конец реплики')
    ap.add_argument('--max-utterance', type=float, default=15.0)
    ap.add_argument('--dialogue-out', help='JSON с репликами диалога')
    ap.add_argument('--no-tts', action='store_true')
    ap.add_argument('--no-translate', action='store_true', help='только глоссы, без LLM')
    ap.add_argument('--show', action='store_true', help='окно при разборе видео')
    ap.add_argument('--mic', action='store_true', help='слушать слышащего микрофоном')
    ap.add_argument('--whisper', default='small')
    ap.add_argument('--hearing', help='реплика слышащего до начала записи (контекст перевода)')
    ap.add_argument('--decoder', choices=('greedy', 'beam'), default='beam')
    ap.add_argument('--beam', type=int, default=16)
    ap.add_argument('--lm-weight', type=float, default=0.1,
                    help='вес LM глоссов в beam search (0 — без LM)')
    ap.add_argument('--word-bonus', type=float, default=2.0,
                    help='бонус за каждое слово: без него LM схлопывает короткие реплики')
    ap.add_argument('--smooth', type=int, default=1,
                    help='окно сглаживания логитов, кадры (1 — выключено: модель даёт пики в 1 кадр)')
    ap.add_argument('--variants', choices=('chains', 'lattice', 'none'), default='chains',
                    help='что из n-best видит переводчик: цепочки, варианты по местам или ничего')
    ap.add_argument('--nbest', type=int, default=8, help='сколько гипотез отдавать переводчику')
    a = ap.parse_args()
    if not a.video and a.camera is None:
        sys.exit('нужен --video или --camera')

    dialogue = Dialogue(a.dialogue_out)
    if a.hearing:
        dialogue.add('hearing', a.hearing)
    rec = Recognizer(a.model, decoder=a.decoder, beam=a.beam, lm_weight=a.lm_weight,
                     word_bonus=a.word_bonus, smooth=a.smooth,
                     nbest=a.nbest if a.variants != 'none' else 1)
    worker = Worker(dialogue, tts=not a.no_tts, translator=not a.no_translate,
                    variants=a.variants)
    mic = None
    if a.mic:
        mic = Mic(dialogue, a.whisper)
        mic.start()
    try:
        if a.video:
            process_video(a.video, rec, dialogue, worker, pause=a.pause,
                          max_len=a.max_utterance, show=a.show)
        else:
            live(a.camera, rec, dialogue, worker, pause=a.pause,
                 max_len=a.max_utterance)
    finally:
        worker.stop = True
        if a.dialogue_out:
            dialogue.save()


if __name__ == '__main__':
    main()
