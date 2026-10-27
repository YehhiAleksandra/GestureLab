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


def test_summarize_seen_and_unseen_words():
    rows = [{'glosses': 0, 'hits': 0, 'total': 2, 'seen_hits': 0, 'seen_total': 1},
            {'glosses': 2, 'hits': 1, 'total': 2, 'seen_hits': 1, 'seen_total': 2}]
    s = fp.summarize(rows)
    assert s['empty'] == 0.5 and s['word_hits'] == 1 and s['word_tokens'] == 4
    assert s['seen_recall'] == round(1 / 3, 3) and s['phrase_all_words'] == 0
