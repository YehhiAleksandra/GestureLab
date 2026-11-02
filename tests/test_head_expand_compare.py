"""Тесты сравнения двух прогонов по клипам: покрытие, расшифровка рядов, бутстрап на общих."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from head_expand_compare import compare, pick, selftest  # noqa: E402


def _row(i, hits, meaning):
    return {'id': i, 'hits_canon': hits, 'total': 2, 'meaning': meaning, 'chrf': meaning}


def test_selftest_passes():
    assert selftest() == 0


def test_pick_indexes_rows_by_clip_id():
    doc = {'tool': 'meaning_compare', 'rows': {'1': [_row('a', 1, 0), _row('b', 0, 2)]}}
    assert set(pick(doc, 1)) == {'a', 'b'}


def test_pick_refuses_unknown_config_and_duplicate_ids():
    with pytest.raises(SystemExit):
        pick({'tool': 'x', 'rows': {'0': []}}, 3)
    with pytest.raises(SystemExit):
        pick({'tool': 'x', 'rows': {'0': [_row('a', 1, 0), _row('a', 0, 0)]}}, 0)


def test_compare_separates_coverage_from_quality():
    """Клипы, которых у старой головы не было, — покрытие, а не качество: в бутстрап они не идут."""
    old = {i: _row(i, 0, 0) for i in 'abcd'}
    new = {**{i: _row(i, 2, 1) for i in 'abcd'}, 'x': _row('x', 1, 0)}
    r = compare(old, new, ['meaning'])
    assert (r['common'], r['unlocked']) == (4, 1)
    assert r['hits_old'] == 0 and r['hits_new'] == 8, r
    assert r['words_common'] == 8 and r['unlock_word_hits'] == 1, r
    assert r['vs_old']['meaning']['wins'] == 4 and r['vs_old']['meaning']['losses'] == 0, r


def test_compare_treats_empty_translation_as_zero():
    """Ряд без перевода (None) судится нулём — иначе пустота осталась бы вне сравнения."""
    old = {'a': {'id': 'a', 'hits_canon': 0, 'total': 1, 'meaning': None, 'chrf': None}}
    new = {'a': {'id': 'a', 'hits_canon': 1, 'total': 1, 'meaning': 2, 'chrf': 30.0}}
    r = compare(old, new, ['meaning', 'chrf'])
    assert r['vs_old']['meaning']['mean_diff'] == 2.0, r
    assert r['vs_old']['chrf']['mean_diff'] == 30.0, r
