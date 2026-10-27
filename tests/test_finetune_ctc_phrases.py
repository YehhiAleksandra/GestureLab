import json

import numpy as np

from tools import finetune_ctc_phrases as fp


def test_folds_cover_every_clip_once():
    parts = fp.folds(10, 5, 0)
    assert sorted(i for f in parts for i in f) == list(range(10))
    assert all(len(f) == 2 for f in parts)
    assert parts == fp.folds(10, 5, 0)
    assert parts != fp.folds(10, 5, 1)


def test_matched_counts_synonyms_as_words():
    assert fp.matched(['мать', 'я'], 'Мама я тебя') == {'я': 1}
    assert fp.matched(['там/туда'], 'туда') == {'туда': 1}
    assert fp.matched([], 'мама') == {}


def test_load_phrase_clips_uses_longest_span_and_skips_short(tmp_path):
    inv = {'phrases': [{'id': 'a', 'phrase': 'Мать я', 'duration': 1.0},
                       {'id': 'b', 'phrase': 'Мать я я', 'duration': 0.2}]}
    (tmp_path / 'inv.json').write_text(json.dumps(inv), encoding='utf-8')
    np.savez_compressed(tmp_path / 'a.npz', frames=np.full((20, 59, 3), 0.5, np.float32),
                        fps=25.0, phrase='Мать я')
    # 4 кадра → 2 выхода CTC, а меток 3 + повтор = 4: учить нельзя
    np.savez_compressed(tmp_path / 'b.npz', frames=np.full((4, 59, 3), 0.5, np.float32),
                        fps=25.0, phrase='Мать я я')
    clips, skipped = fp.load_phrase_clips(tmp_path, tmp_path / 'inv.json', ['мать', 'я'])
    assert [c['id'] for c in clips] == ['a']
    assert clips[0]['labels'] == [0, 1] and clips[0]['used'] == [0, 20]
    assert skipped == [{'id': 'b', 'reason': 'кадров 4 < меток 3'}]


def test_ctc_align_finds_word_borders():
    logp = np.full((12, 3), -5.0)
    logp[:4, 1] = 0.0    # метка 0
    logp[4:6, 0] = 0.0   # blank
    logp[6:10, 2] = 0.0  # метка 1
    logp[10:, 0] = 0.0
    spans = fp.ctc_align(logp, [0, 1])
    assert spans == [(0, 0, 4), (1, 6, 10)]


def test_ctc_align_handles_repeated_labels():
    logp = np.full((14, 2), -5.0)
    logp[1:4, 1] = 0.0
    logp[5:8, 0] = 0.0
    logp[9:12, 1] = 0.0
    spans = fp.ctc_align(logp, [0, 0])
    assert [s[0] for s in spans] == [0, 0] and spans[0][2] <= spans[1][1]


def test_recombine_concatenates_segments_with_gaps():
    seg = np.zeros((8, 59, 3), np.float32)
    pool = {3: [(seg, 'a')], 7: [(seg + 1, 'b')]}
    stream, labels = fp.recombine(pool, np.random.default_rng(0), 2, 2)
    assert len(labels) == 2 and all(l in (3, 7) for l in labels)
    assert stream.shape[1:] == (59, 3)
    assert 14 <= len(stream) <= 16  # 6 + 6 кадров отрезков + пауза 2–4


def test_mix_phrase_recombine_branch_keeps_labels():
    pool = {0: [(np.zeros((8, 59, 3), np.float32), 'a')]}
    s, labels = fp.mix_phrase({}, [], [], [], np.random.default_rng(0), 0.0, 0.0, 0.0,
                              pool=pool, recombine_frac=1.0)
    assert set(labels) == {0} and 2 <= len(labels) <= 4
    assert s.shape[1:] == (59, 3)


def test_summarize_seen_and_unseen_words():
    rows = [{'glosses': 0, 'hits': 0, 'total': 2, 'seen_hits': 0, 'seen_total': 1},
            {'glosses': 2, 'hits': 1, 'total': 2, 'seen_hits': 1, 'seen_total': 2}]
    s = fp.summarize(rows)
    assert s['empty'] == 0.5 and s['word_hits'] == 1 and s['word_tokens'] == 4
    assert s['seen_recall'] == round(1 / 3, 3) and s['phrase_all_words'] == 0


def test_stem_keeps_short_words_and_strips_inflection():
    assert fp.stem('автомобилем') == 'автомобил'
    assert fp.stem('дом') == 'дом'
    assert fp.stem('я') == 'я'


def test_canon_map_only_unique_lemma_candidate():
    assert fp.canon_map({'автомобилем', 'автомобиль'}, {'автомобиль'}) == {
        'автомобиль': 'автомобиль', 'автомобилем': 'автомобиль'}
    # два кандидата на один стем — не гадаем
    assert fp.canon_map({'города'}, {'город', 'городу'}) == {}
