"""Synthetic cache checks; no network or landmark extraction."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from live_signs import NPTS
from tools import rt_dictionary as rt


def cache_video(directory, video, word, length, value):
    directory.mkdir(parents=True, exist_ok=True)
    frames = np.full((length, NPTS, 3), value, dtype=np.float32)
    np.savez_compressed(directory / f'{video}.npz', frames=frames, word=word)
    return frames


def test_limit_does_not_merge_unselected_cache(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    cache = work / 'npz'
    cache_video(cache, 'selected', 'да', 4, 1)
    cache_video(cache, 'unselected', 'нет', 5, 2)
    out = tmp_path / 'merged.npz'
    monkeypatch.setattr(rt, 'WORK', work)
    monkeypatch.setattr(rt, 'list_videos', lambda: [('selected', 'да'), ('unselected', 'нет')])
    monkeypatch.setattr(rt, 'ProcessPoolExecutor', ThreadPoolExecutor)
    selected = []

    def cached_extract(vid, word, out_dir):
        selected.append(vid)
        return vid, 'cached'

    monkeypatch.setattr(rt, 'extract_landmarks', cached_extract)
    monkeypatch.setattr(rt.sys, 'argv', ['rt_dictionary', '--limit', '1', '--out', str(out)])
    rt.main()
    assert selected == ['selected']
    with np.load(out) as data:
        assert data['words'].tolist() == ['да']
        assert data['lens'].tolist() == [4]


def test_selected_merge_ignores_unrelated_cache(tmp_path):
    cache = tmp_path / 'cache'
    expected = cache_video(cache, 'chosen', 'да', 4, 1)
    # A foreign archive must not even be read.
    np.savez_compressed(cache / 'unrelated.npz', other=np.array([0]))
    out = tmp_path / 'merged.npz'
    assert rt.merge(cache, out, video_ids=['chosen']) == 1
    with np.load(out) as data:
        assert data['video'].tolist() == ['chosen']
        assert data['words'].tolist() == ['да']
        assert data['lens'].tolist() == [4]
        np.testing.assert_array_equal(data['frames'], expected)


def test_merge_only_all_cache_and_provenance_alignment(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    cache = work / 'npz'
    second = cache_video(cache, 'z_video', 'нет', 6, 2)
    first = cache_video(cache, 'a_video', 'да', 4, 1)
    out = tmp_path / 'merged.npz'
    monkeypatch.setattr(rt, 'WORK', work)

    def unexpected(*args, **kwargs):
        pytest.fail('merge-only must not list videos or start extraction')

    monkeypatch.setattr(rt, 'list_videos', unexpected)
    monkeypatch.setattr(rt, 'ProcessPoolExecutor', unexpected)
    monkeypatch.setattr(rt.sys, 'argv', [
        'rt_dictionary', '--merge-only', '--limit', '1', '--out', str(out),
    ])
    rt.main()
    with np.load(out) as data:
        assert set(data.files) == {'frames', 'lens', 'words', 'video'}
        assert data['video'].tolist() == ['a_video', 'z_video']
        assert data['words'].tolist() == ['да', 'нет']
        assert data['lens'].tolist() == [4, 6]
        np.testing.assert_array_equal(data['frames'], np.concatenate([first, second]))


def test_empty_cache_returns_zero_without_output(tmp_path):
    out = tmp_path / 'output' / 'merged.npz'
    assert rt.merge(tmp_path / 'cache', out) == 0
    assert not out.exists()
    assert not out.parent.exists()


@pytest.mark.parametrize('video_ids', [[], ['missing']])
def test_empty_selection_returns_zero_without_output(tmp_path, video_ids):
    cache = tmp_path / 'cache'
    cache_video(cache, 'unselected', 'нет', 4, 2)
    out = tmp_path / 'merged.npz'
    assert rt.merge(cache, out, video_ids=video_ids) == 0
    assert not out.exists()
