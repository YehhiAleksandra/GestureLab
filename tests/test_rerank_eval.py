from collections import Counter

from tools import fusion_translate as ft
from tools import rerank_eval as re_


def test_chain_bag_takes_max_multiplicity_not_sum():
    ch = [['мать', 'я'], ['мать', 'мать', 'я'], ['нога']]
    assert re_.chain_bag(ch, 2) == {'мать': 2, 'я': 1}
    assert re_.chain_bag(ch, 3) == {'мать': 2, 'я': 1, 'нога': 1}


def test_votes_bag_needs_min_votes():
    ch = [['а'], ['а', 'б'], ['а', 'б'], ['в']]
    assert re_.votes_bag(ch, 4, 2) == {'а': 3, 'б': 2}
    assert re_.votes_bag(ch, 4, 3) == {'а': 3}


def test_bag_hits_counts_with_multiplicity():
    assert re_.bag_hits(Counter({'вас': 2, 'я': 1}), ['вас', 'вас']) == 2
    assert re_.bag_hits(Counter({'вас': 1}), ['вас', 'вас']) == 1


def test_rerank_prefers_affinity_without_lambda_and_rank_with_big_lambda():
    ch = [['мать', 'я'], ['мать', 'мать', 'я'], ['нога']]
    affs = {'мать': 0.9, 'нога': 0.8, 'я': 0.5}
    assert re_.rerank(ch, affs, 0.5, 0.0) == 2
    assert re_.rerank(ch, affs, 0.5, 1.0) == 0


def test_word_affinity_uses_max_over_synonyms_and_default_for_unknown():
    affs = {'мать': 0.9, 'нога': 0.8, 'я': 0.5}
    assert re_.word_affinity(['там/туда'], {'туда': 0.7}, 0.4) == 0.7
    assert re_.word_affinity(['нет'], affs, 0.4) == 0.4


def test_merge_bags_takes_max():
    merged = re_.merge_bags(Counter({'а': 2}), Counter({'а': 1, 'б': 1}))
    assert merged == {'а': 2, 'б': 1}


def test_fusion_chains_appends_candidates_as_one_pseudo_chain():
    assert ft.fusion_chains([['а']], [('в', 0.9), ('б', 0.8)]) == [['а'], ['в', 'б']]
    assert ft.fusion_chains([['а']], []) == [['а']]


def test_summarize_counts_meaning_and_exact():
    rows = [{'a_meaning': 1, 'b_meaning': 2, 'a_chrf': 10.0, 'b_chrf': 30.0,
             'a_exact': False, 'b_exact': True}]
    s = ft.summarize(rows, 'b')
    assert s['meaning_ge1'] == 1.0 and s['meaning_2'] == 1.0
    assert s['chrf'] == 30.0 and s['exact'] == 1