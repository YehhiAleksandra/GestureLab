import json

from dialogue_state import (DIALOGUE, NO_QUESTIONS, Dialogue, clean_reply, refusal_like,
                            render_glosses, translate_in_context)


class FakeTranslator:
    def __init__(self, answers=None):
        self.calls = []
        self.answers = list(answers or [])

    def __call__(self, glosses, system=None, query=None):
        self.calls.append(('plain', glosses, system, query))
        return self.answers.pop(0) if self.answers else 'без контекста'

    def examples(self, glosses):
        self.calls.append(('examples', glosses))
        return [{'rsl': 'привет', 'russian': 'Привет.'}]

    def chat(self, messages, max_tokens, model=None):
        self.calls.append(('chat', messages, max_tokens))
        return self.answers.pop(0) if self.answers else 'С контекстом.'


def test_clean_reply_strips_list_markers():
    """B42: на одну реплику LLM иногда отвечает «1. Мама…»; перевод должен быть строкой."""
    assert clean_reply('1. Мама, у меня есть их цветовой оттенок.') == \
        'Мама, у меня есть их цветовой оттенок.'
    assert clean_reply('1) Да.\n2) Нет.') == 'Да. Нет.'
    assert clean_reply('- Привет.') == 'Привет.'
    assert clean_reply('  \n ') == ''


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
    assert fake.calls == [('plain', 'мама', 'СИСТЕМА', 'мама')]


def test_candidates_become_bracketed_query():
    fake = FakeTranslator()
    cands = [['мама', 'папа'], ['мыть', 'мыла']]
    translate_in_context(fake, [('мама', 0.9), ('мыть', 0.7)], [], 'СИСТЕМА',
                         candidates=cands)
    kind, flat, system, query = fake.calls[0]
    assert kind == 'plain' and system == 'СИСТЕМА'
    assert flat == 'мама папа мыть мыла'
    assert query == '[мама | папа] [мыть | мыла]'


def test_candidates_with_history_use_flat_glosses_for_examples():
    fake = FakeTranslator()
    history = [{'speaker': 'hearing', 'text': 'Что делаешь?'}]
    translate_in_context(fake, [('мама', 0.9)], history, 'СИСТЕМА',
                         candidates=[['мама', 'папа']])
    kinds = [c[0] for c in fake.calls]
    assert kinds == ['examples', 'chat']
    assert fake.calls[0][1] == 'мама папа'
    assert 'Текущие глоссы: [мама | папа]' in fake.calls[1][1][-1]['content']


def test_chains_become_numbered_query():
    import pytest

    pytest.importorskip('huggingface_hub')
    fake = FakeTranslator()
    chains = [['мама', 'мыть'], ['мама', 'мыть', 'посуда']]
    translate_in_context(fake, [('мама', 0.9)], [], 'СИСТЕМА', chains=chains)
    kind, flat, system, query = fake.calls[0]
    assert kind == 'plain' and system == 'СИСТЕМА'
    assert flat == 'мама мыть мама мыть посуда'
    assert query == '1. мама мыть\n2. мама мыть посуда'


def test_empty_glosses_do_not_reach_llm():
    fake = FakeTranslator()
    chains = [['мама', 'мыть'], ['мама']]
    assert translate_in_context(fake, [], [], 'СИСТЕМА', chains=chains) == ''
    assert translate_in_context(fake, [], [{'speaker': 'deaf', 'text': 'x'}], 'СИСТЕМА',
                                candidates=[['мама', 'папа']]) == ''
    assert fake.calls == []


def test_history_goes_into_prompt_with_speakers():
    fake = FakeTranslator()
    history = [{'speaker': 'hearing', 'text': 'Ты придёшь?'},
               {'speaker': 'deaf', 'text': 'Приду.'}]
    assert translate_in_context(fake, [('завтра', 0.8)], history, 'СИСТЕМА', shots=6) == 'С контекстом.'
    kind, messages, max_tokens = fake.calls[1]
    assert kind == 'chat' and max_tokens == 80
    assert messages[0] == {'role': 'system', 'content': 'СИСТЕМА' + DIALOGUE}
    assert messages[1] == {'role': 'user', 'content': 'привет'}
    assert messages[2] == {'role': 'assistant', 'content': 'Привет.'}
    tail = messages[-1]['content']
    assert 'Слышащий: Ты придёшь?' in tail and 'Глухой (перевод): Приду.' in tail
    assert 'Текущие глоссы: завтра' in tail


# —— B71: переводчик на несвязной цепочке переспрашивает вместо перевода (этап 34) ——

REFUSAL = ('Пожалуйста, уточните контекст или предоставьте глоссы в другом формате. '
           'Данный список выглядит как набор повторяющихся служебных знаков')


def test_refusal_like_finds_meta_answer_but_not_a_real_phrase():
    assert refusal_like(REFUSAL)
    assert refusal_like('Поскольку предоставленные глоссы состоят из одного повторяющегося '
                        'слова «где», они не складываются во фразу.')
    # настоящий перевод, где «уточнить» — часть фразы, а не просьба к входу
    assert not refusal_like('Уточните расписание.')
    assert not refusal_like('Мне невозможно уехать.')
    assert not refusal_like('')


def test_render_glosses_collapses_adjacent_repeats():
    assert render_glosses([('я', 0.9), ('я', 0.8), ('вы', 0.7)]) == 'Я вы.'
    assert render_glosses([('порядок', 0.9), ('быть', 0.8)]) == 'Порядок быть.'
    assert render_glosses([]) == ''


def test_refusal_retried_once_with_stricter_system():
    fake = FakeTranslator([REFUSAL, 'Я в порядке, спасибо.'])
    out = translate_in_context(fake, [('я', 0.9), ('спасибо', 0.8)], [], 'СИСТЕМА')
    assert out == 'Я в порядке, спасибо.'
    assert [c[2] for c in fake.calls] == ['СИСТЕМА', 'СИСТЕМА' + NO_QUESTIONS]


def test_persistent_refusal_ends_with_literal_glosses():
    fake = FakeTranslator([REFUSAL, REFUSAL])
    out = translate_in_context(fake, [('я', 0.9), ('что', 0.5), ('я', 0.6), ('вы', 0.4)], [], 'СИСТЕМА')
    assert out == 'Я что я вы.'
    assert len(fake.calls) == 2
    assert not refusal_like(out)


def test_empty_answer_is_treated_like_refusal():
    """Smoke-прогон этапа 35: на «Увидимся попозже» модель ответила пустотой — то же окном."""
    fake = FakeTranslator(['', ''])
    assert translate_in_context(fake, [('увидимся', 0.9), ('позже', 0.8)], [], 'СИСТЕМА') == \
        'Увидимся позже.'
    fake2 = FakeTranslator(['', 'Увидимся позже.'])
    assert translate_in_context(fake2, [('у', 0.9)], [], 'СИСТЕМА') == 'Увидимся позже.'


def test_refusal_gate_applies_with_history_too():
    fake = FakeTranslator([REFUSAL, 'Приду завтра.'])
    history = [{'speaker': 'hearing', 'text': 'Когда?'}]
    out = translate_in_context(fake, [('завтра', 0.8)], history, 'СИСТЕМА')
    assert out == 'Приду завтра.'
    assert fake.calls[1][1][0]['content'] == 'СИСТЕМА' + DIALOGUE
    assert fake.calls[2][1][0]['content'] == 'СИСТЕМА' + DIALOGUE + NO_QUESTIONS
    assert fake.calls[2][1][1:] == fake.calls[1][1][1:]


def test_empty_top1_still_never_reaches_llm():
    fake = FakeTranslator([REFUSAL])
    assert translate_in_context(fake, [], [], 'СИСТЕМА', chains=[['мама']]) == ''
    assert fake.calls == []
