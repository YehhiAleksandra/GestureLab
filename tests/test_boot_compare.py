"""Тесты парного бутстрапа декодеров: он решает, является ли выбор конфига фактом."""
from tools.boot_compare import paired_bootstrap, per_clip


def test_per_clip_sums_repeats_and_keys_by_fold():
    rows = [{'fold': 0, 'id': 'a', 'hits_canon': 2},
            {'fold': 0, 'id': 'a', 'hits_canon': 1},
            {'fold': 1, 'id': 'a', 'hits_canon': 5}]
    assert per_clip(rows) == {(0, 'a'): 3, (1, 'a'): 5}


def test_bootstrap_only_uses_common_clips():
    a = {(0, 'x'): 3, (0, 'y'): 0}
    b = {(0, 'x'): 2, (0, 'y'): 1, (0, 'only_b'): 9}
    r = paired_bootstrap(a, b, n=500)
    assert r['clips'] == 2 and r['a_wins'] == 1 and r['b_wins'] == 1
    assert r['mean_diff'] == 0.0 and r['lo95'] <= 0 <= r['hi95']


def test_steady_advantage_is_distinguishable():
    a = {(0, f'c{i}'): 2 for i in range(40)}
    b = {(0, f'c{i}'): 1 for i in range(40)}
    r = paired_bootstrap(a, b, n=2000)
    assert r['mean_diff'] == 1.0 and r['lo95'] > 0 and r['p_a_better'] == 1.0


def test_bootstrap_is_deterministic_per_seed():
    a = {(0, f'c{i}'): int(i % 3) for i in range(30)}
    b = {(0, f'c{i}'): int(i % 2) for i in range(30)}
    assert paired_bootstrap(a, b, n=800) == paired_bootstrap(a, b, n=800)
