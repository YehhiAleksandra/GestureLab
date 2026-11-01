import json

import numpy as np

from tools import centroid_eval as ce


def test_build_centroids_averages_clips_of_same_word():
    e1, e2, e3 = np.eye(3, dtype=np.float32)
    cents = ce.build_centroids([('рука', e1), ('рука', e2), ('нога', e3)])
    assert set(cents) == {'рука', 'нога'}
    assert np.allclose(cents['рука'], (e1 + e2) / 2)
    assert np.allclose(cents['нога'], e3)


def test_rank_returns_nearest_first():
    e1, e2, e3 = np.eye(3, dtype=np.float32)
    cents = ce.build_centroids([('а', e1), ('б', e2), ('в', e3)])
    cw = sorted(cents)
    cm = np.stack([cents[w] for w in cw])
    assert ce.rank(cw, cm, e1, 2) == ['а', 'б']
    assert ce.rank(cw, cm, e2 + 0.1 * e3, 3) == ['б', 'в', 'а']


def test_match_bag_counts_with_multiplicity():
    assert ce.match_bag(['вас', 'я'], ['вас', 'вас', 'я']) == 2
    assert ce.match_bag(['вас', 'вас'], ['вас']) == 1
    assert ce.match_bag([], ['вас']) == 0


def test_load_dict_clips_slices_frames_by_lens(tmp_path):
    frames = np.arange(5 * 59 * 3, dtype=np.float32).reshape(5, 59, 3)
    np.savez_compressed(tmp_path / 'd.npz', frames=frames,
                        lens=np.array([2, 3]), words=np.array(['Мать', 'Улица']))
    clips = ce.load_dict_clips(tmp_path / 'd.npz')
    assert [w for w, _ in clips] == ['мать', 'улица']
    assert len(clips[0][1]) == 2 and len(clips[1][1]) == 3
    assert np.allclose(clips[1][1], frames[2:])


def test_load_phrase_spans_uses_longest_span(tmp_path):
    inv = {'phrases': [{'id': 'a', 'phrase': 'Мать я', 'duration': 1.0}]}
    (tmp_path / 'inv.json').write_text(json.dumps(inv), encoding='utf-8')
    frames = np.full((20, 59, 3), 0.5, np.float32)
    np.savez_compressed(tmp_path / 'a.npz', frames=frames, fps=25.0, phrase='Мать я')
    clips, skipped = ce.load_phrase_spans(tmp_path, tmp_path / 'inv.json')
    assert not skipped
    assert clips[0]['ref'] == ['мать', 'я'] and clips[0]['used'] == [0, 20]


def test_make_chunks_covers_long_clip_with_overlap():
    long = np.zeros((40, 59, 3), np.float32)
    parts = ce.make_chunks(long, 32, 16)
    assert len(parts) == 2 and len(parts[1]) == 32
    assert np.shares_memory(parts[1][0], long[8])
    assert len(ce.make_chunks(long[:10], 32, 16)) == 1
    assert len(ce.make_chunks(long, 40, 16)) == 1
    base = np.zeros((50, 59, 3), np.float32)
    chunks48 = ce.make_chunks(base, 48, 16)
    assert len(chunks48) == 2 and np.shares_memory(chunks48[1][0], base[2])