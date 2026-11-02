"""Тесты сквозного замера смысла: разбор конфигов, бутстрап и агрегат без LLM."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from meaning_compare import aggregate, boot_diff, parse_decoders, selftest  # noqa: E402


def test_selftest_passes():
    assert selftest() == 0


def test_parse_decoders_zeroes_weight_without_lm():
    cfgs = parse_decoders('none:0.1:4.0:3,corpus:0.1:2.0:1')
    assert cfgs[0]['lm_weight'] == 0.0, 'без LM вес не значит ничего'
    assert cfgs[1]['lm_weight'] == 0.1 and cfgs[1]['smooth'] == 1, cfgs


def test_parse_decoders_rejects_garbage():
    with pytest.raises(SystemExit):
        parse_decoders('corpus:0.1:2.0')


def test_boot_diff_is_deterministic_and_paired():
    a = [2, 0, 1, 0, 2, 0, 0, 0]
    b = [0, 0, 0, 0, 1, 0, 0, 0]
    first, again = boot_diff(a, b), boot_diff(a, b)
    assert first == again, 'один seed — один ЦИ'
    assert first['a_wins'] == 3 and first['b_wins'] == 0
    same = boot_diff(a, a)
    assert same['mean_diff'] == 0.0 and not same['distinguishable'], same


def test_boot_diff_small_sample_is_not_distinguished():
    # два клипа, одно попадание: разница есть, но ЦИ должен содержать 0
    d = boot_diff([1, 0], [0, 0])
    assert d['mean_diff'] == 0.5 and not d['distinguishable'], d


def test_aggregate_counts_empty_and_meaning():
    rows = [{'glosses': 0, 'hits_canon': 0, 'total': 2, 'meaning': 0, 'chrf': 0.0},
            {'glosses': 4, 'hits_canon': 2, 'total': 3, 'meaning': 1, 'chrf': 30.0},
            {'glosses': 1, 'hits_canon': 0, 'total': 1, 'meaning': 2, 'chrf': 60.0,
             'error': 'TimeoutError'}]
    s = aggregate(rows, None)
    assert s['clips'] == 3 and s['empty'] == round(1 / 3, 3), s
    assert s['meaning_ge1'] == round(2 / 3, 3) and s['meaning_2'] == round(1 / 3, 3), s
    assert s['llm_errors'] == 1, s
    assert s['word_hits_canon'] == 2 and s['word_tokens'] == 6, s
