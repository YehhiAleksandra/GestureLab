import json
from pathlib import Path

import numpy as np
import pytest

from tools.sign_trainer import (attempt_result, hands_ratio, norm, rows_for, trim,
                               verdict)

REPO = Path(__file__).resolve().parent.parent


class FakeRec:
    """Подставной распознаватель: decode возвращает заданные глоссы."""

    def __init__(self, glosses=()):
        self.glosses = list(glosses)

    def decode(self, frames):
        return self.glosses, None


def stats():
    return {'attempts': 0, 'match': 0, 'near': 0, 'miss': 0, 'controls': 0, 'halluc': 0}


def attempt(n=30, mode='repeat'):
    """Буфер записи: n кадров, в каждом видна кисть."""
    frame = np.zeros((59, 3), np.float32)
    frame[0] = 0.5
    return {'mode': mode, 'buf': [frame.copy() for _ in range(n)], 'started': 0.0}


def seq(frames_with_hands):
    """[10, 59, 3] с кистью только в указанных кадрах."""
    arr = np.full((10, 59, 3), np.nan, np.float32)
    for i in frames_with_hands:
        arr[i, 0, 0] = 0.5
    return arr


def test_verdict_separates_match_near_miss_empty():
    assert verdict('вода', [('вода', 0.9)])[0] == 'match'
    assert verdict('Вода', [('вода', 0.5)])[0] == 'match'
    assert verdict('елка', [('ёлка', 0.5)])[0] == 'match'
    assert verdict('вода', [('еда', 0.9)], [['еда'], ['вода']])[0] == 'near'
    assert verdict('вода', [('еда', 0.9)], [['еда']])[0] == 'miss'
    assert verdict('вода', [])[0] == 'empty'


def test_norm_collapses_case_and_yo():
    assert norm('  ВОДА ') == 'вода'
    assert norm('Ёлка') == 'елка'


def test_trim_cuts_frames_without_hands():
    assert len(trim(seq([3, 4, 5, 6]))) == 4
    assert hands_ratio(trim(seq([3, 4, 5, 6]))) == 1.0
    assert len(trim(seq(range(10)))) == 10
    assert len(trim(seq([]))) == 0


def test_attempt_result_counts_match_and_miss():
    st = stats()
    entry = {'word': 'вода'}
    kind, _text, glosses = attempt_result(entry, attempt(), 25.0,
                                          FakeRec([('вода', 0.9)]), st)
    assert kind == 'match' and glosses[0][0] == 'вода'
    assert st['attempts'] == 1 and st['match'] == 1

    kind, _text, _g = attempt_result(entry, attempt(), 25.0,
                                     FakeRec([('еда', 0.9)]), st)
    assert kind == 'miss' and st['attempts'] == 2 and st['miss'] == 1


def test_attempt_result_control_separates_silence_from_hallucination():
    st = stats()
    entry = {'word': 'вода'}
    kind, _text, _g = attempt_result(entry, attempt(mode='control'), 25.0,
                                     FakeRec([]), st)
    assert kind == 'match' and st['controls'] == 1 and st['attempts'] == 0

    kind, text, _g = attempt_result(entry, attempt(mode='control'), 25.0,
                                    FakeRec([('еда', 0.9)]), st)
    assert kind == 'miss' and 'галлюцинация' in text
    assert st['halluc'] == 1 and st['controls'] == 1


def test_attempt_result_ignores_empty_recording():
    st = stats()
    kind, _text, glosses = attempt_result({'word': 'вода'}, attempt(n=0), 25.0,
                                          FakeRec([('вода', 0.9)]), st)
    assert kind == 'empty' and glosses == [] and st == stats()


def test_rows_show_target_model_and_tally():
    stats = {'attempts': 2, 'match': 1, 'near': 0, 'miss': 1, 'controls': 1, 'halluc': 0}
    rows = rows_for('вода', 0.9, 'match', 'СОВПАЛО', [('вода', 0.9)], stats)
    assert 'вода' in rows[0][1] and '0.90' in rows[0][1]
    assert rows[1][1] == 'СОВПАЛО'
    assert 'вода (0.90)' in rows[2][1]
    assert '1/2' in rows[3][1] and 'галлюцинаций 0' in rows[3][1]


def test_roster_is_unique_and_within_probability():
    path = REPO / 'reports' / 'evidence' / 'trainer_roster.json'
    if not path.exists():
        pytest.skip('словарь тренажёра ещё не собран')
    roster = json.loads(path.read_text(encoding='utf-8'))
    words = [e['word'] for e in roster['words']]
    ids = [e['id'] for e in roster['words']]
    assert words, 'словарь пуст'
    assert len(set(words)) == len(words) == len(set(ids))
    assert all(0.0 <= e['p'] <= 1.0 for e in roster['words'])
