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
