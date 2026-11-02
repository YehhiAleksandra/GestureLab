import numpy as np

from tools.finetune_dict import build_dict_pool, load_extra_dict, mix_dict, recombine_dict


def test_build_dict_pool_normalizes_and_maps_by_class():
    names = ['мать', 'нога', 'бобр']
    dw = [('мать', np.zeros((10, 59, 3), np.float32)),
          ('Нога', np.ones((8, 59, 3), np.float32)),
          ('кит', np.full((6, 59, 3), 2.0, np.float32))]
    pool, classes, covered = build_dict_pool(dw, names)
    assert covered == 2 and classes == [0, 1]
    assert len(pool[0]) == 1 and len(pool[1]) == 1


def test_recombine_dict_unique_labels_and_valid_length():
    pool = {c: [np.full((20, 59, 3), c, np.float32)] for c in range(3)}
    rng = np.random.default_rng(0)
    for i in range(20):
        s, labels = recombine_dict(pool, [0, 1, 2], np.random.default_rng(i))
        assert 2 <= len(labels) <= 5 and len(set(labels)) == len(labels)
        assert s.ndim == 3 and s.shape[1] == 59 and len(s) >= 2 * len(labels) + 1


def test_recombine_dict_caps_k_by_pool_size():
    pool = {0: [np.zeros((20, 59, 3), np.float32)],
            1: [np.zeros((20, 59, 3), np.float32)]}
    s, labels = recombine_dict(pool, [0, 1], np.random.default_rng(3))
    assert len(labels) == 2


def test_mix_dict_returns_valid_examples_for_all_sources():
    pool = {c: [np.full((20, 59, 3), c, np.float32)] for c in range(3)}
    slovo_pool = {c: [np.full((20, 59, 3), c, np.float32)] for c in range(3)}
    phrases = [(np.zeros((40, 59, 3), np.float32), [0, 1])]
    for i in range(60):
        s, lab = mix_dict(pool, [0, 1, 2], slovo_pool, [0, 1, 2] * 3, [],
                          phrases, np.random.default_rng(i))
        assert s.ndim == 3 and s.shape[1] == 59 and lab


def test_load_extra_dict_cleans_labels_and_filters_junk(tmp_path):
    f = np.zeros((30, 59, 3), np.float32)
    p = tmp_path / 'extra.npz'
    np.savez(p, frames=f, offsets=np.array([0, 10, 20, 30]),
             words=np.array(['- Абрикос', 'Вариант 1', 'абрикос']))
    ex = load_extra_dict(p)
    # «- Абрикос» и «абрикос» сливаются, «Вариант 1» с цифрой отфильтрован
    assert [w for w, _ in ex] == ['абрикос', 'абрикос']


def test_load_extra_dict_caps_clips_per_word(tmp_path):
    f = np.zeros((60, 59, 3), np.float32)
    p = tmp_path / 'extra.npz'
    np.savez(p, frames=f, offsets=np.array([0, 10, 20, 30, 40, 50, 60]),
             words=np.array(['мать'] * 6))
    assert len(load_extra_dict(p, cap=4)) == 4