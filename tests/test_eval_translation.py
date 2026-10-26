import json

import pytest


@pytest.fixture
def ev():
    pytest.importorskip('huggingface_hub')
    from tools import eval_translation
    return eval_translation


def test_load_rows_dedups_by_glosses_and_ref(tmp_path, ev):
    row = {'glosses_ref': 'мама мыть', 'ref': 'Мама мыла.', 'cands': [['мама'], ['мыть']]}
    other = {'glosses_ref': 'пить вода', 'ref': 'Пить воду.', 'cands': [['пить'], ['вода']]}
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps({'rows': [row, other]}, ensure_ascii=False), encoding='utf-8')
    b.write_text(json.dumps({'rows': [row]}, ensure_ascii=False), encoding='utf-8')
    rows = ev.load_rows([a, b])
    assert [r['ref'] for r in rows] == ['Мама мыла.', 'Пить воду.']


def test_load_rows_rejects_rows_without_candidates(tmp_path, ev):
    """B40: старый артефакт хранил перевод под ключом lattice, кандидаты терялись."""
    path = tmp_path / 'old.json'
    path.write_text(json.dumps({'rows': [{'glosses_ref': 'мама мыть', 'ref': 'Мама мыла.',
                                          'lattice': 'Мама мыла.'}]}, ensure_ascii=False),
                    encoding='utf-8')
    with pytest.raises(ValueError, match='candidates'):
        ev.load_rows([path])


def test_lattice_variant_reads_candidates_not_translation(ev):
    class FakeTranslator:
        def __init__(self):
            self.calls = []

        def __call__(self, glosses, system=None, query=None):
            self.calls.append((glosses, system, query))
            return 'x'

    tr = FakeTranslator()
    row = {'glosses_beam': 'мама мыть', 'ref': 'Мама мыла.',
           'cands': [['мама', 'папа'], ['мыть']], 'lattice': 'чужой перевод'}
    ev.translate(tr, row, 'lattice')
    assert tr.calls[0][0] == 'мама папа мыть'
    assert tr.calls[0][2] == '[мама | папа] [мыть]'


def test_lattice_probs_variant_puts_confidence_next_to_each_word(ev):
    class FakeTranslator:
        def __init__(self):
            self.calls = []

        def __call__(self, glosses, system=None, query=None):
            self.calls.append((glosses, system, query))
            return 'x'

    tr = FakeTranslator()
    row = {'glosses_beam': 'мама мыть', 'ref': 'Мама мыла.',
           'cands': [['мама', 'папа'], ['мыть']], 'cands_probs': [[0.7, 0.3], [1.0]]}
    ev.translate(tr, row, 'lattice_probs')
    assert tr.calls[0][2] == '[мама 70% | папа 30%] [мыть 100%]'
    assert 'уверенность' in tr.calls[0][1]


def test_nbest_lines_variant_lists_whole_chains(ev):
    class FakeTranslator:
        def __init__(self):
            self.calls = []

        def __call__(self, glosses, system=None, query=None):
            self.calls.append((glosses, system, query))
            return 'x'

    tr = FakeTranslator()
    row = {'glosses_beam': 'мама мыть', 'ref': 'Мама мыла.',
           'cands': [['мама'], ['мыть']],
           'nbest': [['мама', 'мыть'], ['мама', 'мыть', 'посуда']]}
    ev.translate(tr, row, 'nbest_lines')
    assert tr.calls[0][2] == '1. мама мыть\n2. мама мыть посуда'
    assert 'несколько вариантов' in tr.calls[0][1]


def test_summarize_counts_meaning_and_chrf(ev):
    res = [{'hyp': 'Мама мыла раму.', 'judge': 2},
           {'hyp': 'что-то', 'judge': 1},
           {'hyp': 'нет', 'judge': 0}]
    refs = ['Мама мыла раму.', 'Мама мыла раму.', 'Мама мыла раму.']
    s = ev.summarize(res, refs)
    assert s['meaning_ok'] == 0.333 and s['meaning_gist'] == 0.667
    assert s['judged'] == 3 and s['errors'] == 0 and s['chrf'] is not None


def test_paired_compares_only_judged_rows(ev):
    a = [{'judge': 2}, {'judge': 1}, {'judge': 0}, {'error': 'x'}]
    b = [{'judge': 1}, {'judge': 1}, {'judge': 0}, {'judge': 2}]
    p = ev.paired(a, b, n=200)
    assert (p['wins'], p['losses'], p['ties']) == (1, 0, 2)
    assert p['mean_diff'] == 0.333


def test_pick_index_maps_answer_to_chain_and_falls_back(ev):
    assert ev.pick_index('3', 20) == 2
    assert ev.pick_index('Вариант: 12.', 20) == 11
    assert ev.pick_index('не знаю', 20) is None
    assert ev.pick_index('25', 20) is None          # вне списка — не выдумывать цепочку
    assert ev.pick_index('', 20) is None


def test_cot_text_strips_format_and_keeps_plain_answer(ev):
    assert ev.cot_text('Вариант: 2. Перевод: «Мама мыла посуду.»') == 'Мама мыла посуду.'
    assert ev.cot_text('Мама мыла посуду.') == 'Мама мыла посуду.'


def test_select_variant_asks_for_number_then_translates_chosen_chain(ev):
    class FakeTranslator:
        def __init__(self):
            self.calls = []

        def chat(self, messages, max_tokens, model=None):
            self.calls.append(('chat', messages, max_tokens))
            return '2'

        def __call__(self, glosses, system=None, query=None):
            self.calls.append(('plain', glosses, system, query))
            return 'Мама мыла посуду.'

    tr = FakeTranslator()
    row = {'glosses_beam': 'мама мыть', 'ref': 'Мама мыла посуду.',
           'cands': [['мама'], ['мыть']],
           'nbest': [['мама', 'мыть'], ['мама', 'мыть', 'посуда']]}
    out = ev.translate(tr, row, 'select_then_translate')
    assert tr.calls[0][0] == 'chat' and 'только номер' in tr.calls[0][1][0]['content']
    assert tr.calls[0][1][1]['content'].startswith('1. мама мыть')
    assert tr.calls[1][1] == 'мама мыть посуда'      # переводим выбранную цепочку
    assert out['pick'] == 1 and out['text'] == 'Мама мыла посуду.'


def test_pick_stats_counts_choice_against_reference(ev):
    rows = [{'glosses_ref': 'мама мыть', 'oracle_exact_nbest': 0},
            {'glosses_ref': 'пить вода', 'oracle_exact_nbest': 1}]
    res = [{'pick': 0, 'chain': ['мама', 'мыть'], 'judge': 2},
           {'pick': 1, 'chain': ['пить', 'вода'], 'judge': 2}]
    ps = ev.pick_stats(res, rows)
    assert ps['pick_asked'] == 2 and ps['pick_top1'] == 0.5
    assert ps['pick_correct_when_present'] == 1.0
