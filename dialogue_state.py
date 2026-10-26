"""Состояние диалога и контекстный промпт (только stdlib — проверяется в CI).

Сам переводчик с камерой (tools/dialogue_translator.py) тянет torch, mediapipe и LLM,
поэтому логика, которую можно проверить без них, живёт здесь.
"""
import json
import threading
from datetime import datetime
from pathlib import Path

DIALOGUE = (' Ниже — предыдущие реплики диалога. Переведи только текущую реплику глухого, '
            'используя контекст для местоимений и пропущенных слов. Новых фактов не добавляй.')


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
    if chains:
        # модуль тянет huggingface_hub: импорт по факту; путь от корня репо — как в тестах
        from tools.gloss_to_text import chain_list
        flat = ' '.join(w for c in chains for w in c)
        query = chain_list(chains)
    elif candidates:
        flat = ' '.join(w for c in candidates for w in c)
        query = ' '.join('[' + ' | '.join(c) + ']' for c in candidates)
    else:
        flat, query = gloss_text, gloss_text
    if not history:
        return translator(flat, system, query)
    ctx = '\n'.join(f"{'Слышащий' if t['speaker'] == 'hearing' else 'Глухой (перевод)'}: {t['text']}"
                    for t in history)
    messages = [{'role': 'system', 'content': system + DIALOGUE}]
    for ex in translator.examples(flat)[:shots]:
        messages += [{'role': 'user', 'content': ex['rsl']},
                     {'role': 'assistant', 'content': ex['russian']}]
    messages.append({'role': 'user',
                     'content': f'Контекст диалога:\n{ctx}\n\nТекущие глоссы: {query}'})
    return translator.chat(messages, 80)
