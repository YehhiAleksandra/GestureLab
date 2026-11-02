"""Тесты словарного разрыва: канон по лемме и цена добавления класса."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from vocab_gap import unlock_curve  # noqa: E402
from finetune_ctc_phrases import canon_map, head_names, lemma_index  # noqa: E402
from finetune_ctc_signer import class_index  # noqa: E402
from vocab_gap import phrase_gap, selftest  # noqa: E402


class StubMorph:
    def __init__(self, lemmas):
        self.lemmas = lemmas

    def parse(self, w):
        outer = self

        class _P:
            normal_form = outer.lemmas.get(w, w)
        return [_P()]


def test_selftest_passes():
    assert selftest() == 0


def test_lemma_canon_unlocks_what_stem_cannot():
    vocab = set(class_index(['дом', 'автомобиль']))
    plain = ['дома', 'автомобиля']
    assert canon_map(plain, vocab) == {'автомобиля': 'автомобиль'}, \
        'грубая основа не режет окончание у короткого слова'
    lem = {'дома': 'дом'}
    assert canon_map(plain, vocab, StubMorph(lem)) == {'автомобиля': 'автомобиль', 'дома': 'дом'}


def test_class_index_reads_both_yo_forms():
    """Класс «всё» ищется и как «все» (тексты фраз режут «ё»), и как «всё» (глоссы корпуса)."""
    idx = class_index(['всё', 'там/туда'])
    assert idx['все'] == idx['всё'] == 0
    assert idx['там'] == idx['туда'] == 1
    assert head_names(['всё']) == {'все', 'всё'}


def test_ambiguous_lemma_is_not_mapped():
    # две строки головы с одной леммой — выбирать не из чего
    assert lemma_index({'дом', 'дома'}, StubMorph({'дома': 'дом'})) == {}
    assert canon_map(['дома'], {'дом', 'дома'}, StubMorph({'дома': 'дом'})).get('дома') in \
        (None, 'дом', 'дома')


@pytest.mark.parametrize('word,lemma', [('дома', 'дом'), ('автомобиля', 'автомобиль'),
                                        ('человека', 'человек'), ('пятью', 'пять')])
def test_real_morphology(word, lemma):
    from finetune_ctc_phrases import load_morph
    assert load_morph().parse(word)[0].normal_form == lemma


def test_unlock_curve_price_per_class():
    blocked = {'a': frozenset(['x']), 'b': frozenset(['x', 'y']), 'c': frozenset(['y'])}
    order, need, curve, goal = unlock_curve(['u'], blocked, 4)
    assert order == ['x', 'y'] and need == [1, 2, 2]
    assert curve[0] == {'classes': 0, 'phases': 1, 'share': 0.25}
    # 40% от 4 фраз = 1.6: одного класса уже хватает (x разблокирует a и b)
    assert goal == 1


def test_phrase_gap_drops_clips_without_cache(tmp_path):
    inv = tmp_path / 'inv.json'
    inv.write_text(json.dumps({'phrases': [{'id': 'a', 'phrase': 'Дом', 'duration': 1.0}]}),
                   encoding='utf-8')
    vocab = {'дом'}
    assert phrase_gap(inv, vocab, {}) == (['a'], {})
    assert phrase_gap(inv, vocab, {}, tmp_path / 'no_cache') == ([], {})
