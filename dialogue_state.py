"""Состояние диалога и контекстный промпт (только stdlib — проверяется в CI).

Сам переводчик с камерой (tools/dialogue_translator.py) тянет torch, mediapipe и LLM,
поэтому логика, которую можно проверить без них, живёт здесь.
"""
import json
import re
import threading
from datetime import datetime
from pathlib import Path

DIALOGUE = (' Ниже — предыдущие реплики диалога. Переведи только текущую реплику глухого, '
            'используя контекст для местоимений и пропущенных слов. Новых фактов не добавляй.')

_LIST = re.compile(r'^\s*(?:\d+[.)]|[-•*])\s+')

# B71: на несвязной цепочке глосс модель отвечает переспрашиванием («Пожалуйста, уточните
# контекст или предоставьте глоссы…») вместо перевода. Детектор строит признак по паре
# «просят вход» + «говорят о самом входе», а не по одному слову: ложный флаг выбросил бы
# настоящий перевод, а пропущенный лишь оставляет поведение как было.
_ASK = ('уточн', 'предостав', 'пришл', 'перешл', 'прошу вас')
_ABOUT_INPUT = ('глосс', 'список', 'формат', 'ввод', 'последовательност', 'набор',
                'текст', 'данный', 'данные')
NO_QUESTIONS = (' Не переспрашивай и не объясняй, что вход неполный: переведи глоссы как есть, '
                'пропустив то, что не читается. Ответ — одна бытовая фраза без пояснений.')


def clean_reply(text):
    """Одна реплика — одна строка: LLM иногда отвечает «1. Мама…» или списком (B42)."""
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    if not lines:
        return ''
    return ' '.join(_LIST.sub('', l).strip() for l in lines)


def refusal_like(text):
    """Ответ не перевод, а просьба дать другой вход."""
    t = str(text).lower()
    return any(a in t for a in _ASK) and any(m in t for m in _ABOUT_INPUT)


def render_glosses(glosses):
    """Буквальное чтение цепочки: склеить top-1, схлопнув соседние повторы.

    Фолбэк для B71 — пустой экран или переспрашивание глухому бесполезны, а слова глосс
    он прочитает сам; сравнение с «глоссы как есть» уже есть базовой линией в оценке.
    """
    out = []
    for g, _ in glosses:
        if not out or out[-1].lower() != str(g).lower():
            out.append(str(g))
    if not out:
        return ''
    text = ' '.join(out)
    return (text[0].upper() + text[1:]).rstrip('.!?) ') + '.'


def _no_refusal(first, retry, glosses):
    """Отказ или пустой ответ → одна повторная попытка со строгим запретом вопросов → цепочка.

    Пустой ответ языковой модель (в smoke-прогоне этапа 35 такие были) — тот же провал,
    что и переспрашивание: глухой видит пустое окно.
    """
    text = clean_reply(first)
    if not text or refusal_like(text):
        text = clean_reply(retry())
    return render_glosses(glosses) if not text or refusal_like(text) else text


class Dialogue:
    def __init__(self, out=None):
        self.turns, self.out = [], out
        self.lock = threading.Lock()

    def add(self, speaker, text, glosses=None, confidence=None, at=None):
        with self.lock:
            self.turns.append({'speaker': speaker, 'text': text, 'glosses': glosses,
                               'confidence': confidence,
                               'time': at if at is not None else datetime.now().isoformat(timespec='seconds')})
            self.save()
            return self.turns[-1]

    def context(self, k=4):
        with self.lock:
            return [t for t in self.turns if t['text']][-k:]

    def save(self):
        if not self.out:
            return
        Path(self.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(self.out) + '.tmp')
        tmp.write_text(json.dumps(self.turns, ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.replace(self.out)


def translate_in_context(translator, glosses, history, system, shots=6, candidates=None,
                         chains=None):
    """Перевод реплики с учётом последних реплик; translator — gloss_to_text.Translator.

    candidates — варианты n-best на каждое место ([['мама', 'папа'], ...]). Тогда в запросе
    каждый жест в квадратных скобках, а few-shot ищется по всем вариантам сразу.
    chains — целые цепочки глоссов из n-best ([['мама', 'мыть'], ...]): в запросе
    нумерованный список, модели видно фразу целиком.
    """
    gloss_text = ' '.join(g for g, _ in glosses)
    if not gloss_text:
        # B50: пустой top-1 — «ничего не распознано». Цепочки n-best тут только вредят:
        # LLM сочиняет перевод по мусору, и замер смысла засчитывает галлюцинацию.
        return ''
    if chains:
        # модуль тянет huggingface_hub: импорт по факту; путь от корня репо — как в тестах
        from tools.gloss_to_text import chain_list
        flat = ' '.join(w for c in chains for w in c)
        query = chain_list(chains, len(chains))
    elif candidates:
        flat = ' '.join(w for c in candidates for w in c)
        query = ' '.join('[' + ' | '.join(c) + ']' for c in candidates)
    else:
        flat, query = gloss_text, gloss_text
    if not history:
        return _no_refusal(translator(flat, system, query),
                           lambda: translator(flat, system + NO_QUESTIONS, query), glosses)
    ctx = '\n'.join(f"{'Слышащий' if t['speaker'] == 'hearing' else 'Глухой (перевод)'}: {t['text']}"
                    for t in history)
    messages = [{'role': 'system', 'content': system + DIALOGUE}]
    for ex in translator.examples(flat)[:shots]:
        messages += [{'role': 'user', 'content': ex['rsl']},
                     {'role': 'assistant', 'content': ex['russian']}]
    messages.append({'role': 'user',
                     'content': f'Контекст диалога:\n{ctx}\n\nТекущие глоссы: {query}'})

    def stricter():
        # дефолтный промпт остаётся дефолтным: строгий запрет вопросов — только второй заход
        head = dict(messages[0], content=messages[0]['content'] + NO_QUESTIONS)
        return translator.chat([head, *messages[1:]], 80)

    return _no_refusal(translator.chat(messages, 80), stricter, glosses)
