import json

from dialogue_state import DIALOGUE, Dialogue, translate_in_context


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def __call__(self, glosses, system=None, query=None):
        self.calls.append(('plain', glosses))
        return 'без контекста'

    def examples(self, glosses):
        return [{'rsl': 'привет', 'russian': 'Привет.'}]

    def chat(self, messages, max_tokens, model=None):
        self.calls.append(('chat', messages, max_tokens))
        return 'С контекстом.'


def test_dialogue_keeps_last_turns_and_saves_atomically(tmp_path):
    out = tmp_path / 'dialog.json'
    d = Dialogue(str(out))
    d.add('hearing', 'Ты придёшь?')
    d.add('deaf', '', glosses=['прийти'], at='0:03')
    d.add('deaf', 'Приду.', glosses=['прийти'], confidence=[0.9], at='0:05')
    assert [t['text'] for t in d.context()] == ['Ты придёшь?', 'Приду.']
    assert d.turns[1]['time'] == '0:03' and d.turns[2]['confidence'] == [0.9]
    assert json.loads(out.read_text(encoding='utf-8')) == d.turns
    assert not (tmp_path / 'dialog.json.tmp').exists()


def test_context_limited_to_k():
    d = Dialogue()
    for i in range(6):
        d.add('deaf', f'реплика {i}')
    assert [t['text'] for t in d.context(k=2)] == ['реплика 4', 'реплика 5']


def test_empty_history_uses_plain_translator():
    fake = FakeTranslator()
    assert translate_in_context(fake, [('мама', 0.9)], [], 'СИСТЕМА') == 'без контекста'
    assert fake.calls == [('plain', 'мама')]


def test_history_goes_into_prompt_with_speakers():
    fake = FakeTranslator()
    history = [{'speaker': 'hearing', 'text': 'Ты придёшь?'},
               {'speaker': 'deaf', 'text': 'Приду.'}]
    assert translate_in_context(fake, [('завтра', 0.8)], history, 'СИСТЕМА', shots=6) == 'С контекстом.'
    kind, messages, max_tokens = fake.calls[0]
    assert kind == 'chat' and max_tokens == 80
    assert messages[0] == {'role': 'system', 'content': 'СИСТЕМА' + DIALOGUE}
    assert messages[1] == {'role': 'user', 'content': 'привет'}
    assert messages[2] == {'role': 'assistant', 'content': 'Привет.'}
    tail = messages[-1]['content']
    assert 'Слышащий: Ты придёшь?' in tail and 'Глухой (перевод): Приду.' in tail
    assert 'Текущие глоссы: завтра' in tail
