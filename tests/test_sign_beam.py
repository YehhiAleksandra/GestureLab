import numpy as np

from sign_beam import GlossLM, ctc_beam_search, nbest_to_lattice, smooth_log_probs


def _logp(rows):
    p = np.full((len(rows), 4), 1e-3)
    for t, (c, v) in enumerate(rows):
        p[t, c] = v
    p /= p.sum(-1, keepdims=True)
    return np.log(p)


def test_smooth_log_probs_keeps_shape_and_normalization():
    logp = _logp([(0, 1.0), (3, 1.0), (2, 1.0), (0, 1.0), (1, 1.0)])
    out = smooth_log_probs(logp, width=3)
    assert out.shape == logp.shape
    assert np.allclose(np.exp(out).sum(-1), 1.0, atol=1e-6)


def test_beam_search_matches_clear_path():
    logp = _logp([(0, 0.9), (2, 0.9), (0, 0.9), (3, 0.9), (0, 0.9)])
    hyps = ctc_beam_search(logp, beam=4, topk=3, nbest=2)
    assert hyps[0][0] == [1, 2]
    assert len(hyps) == 2 and hyps[1][1] <= hyps[0][1]


def test_lm_breaks_tie_between_words():
    logp = np.array([[-20.0, -20.0, -20.0, -20.0],
                     [-20.0, -0.7, -0.7, -20.0],
                     [-20.0, -20.0, -20.0, -20.0]])
    lm = GlossLM([[0, 1], [0, 1], [0, 2]], n_classes=3)
    plain = ctc_beam_search(logp, beam=4, topk=3, nbest=1)[0][0]
    fused = ctc_beam_search(logp, beam=4, topk=3, nbest=1, lm=lm, lm_weight=2.0)[0][0]
    assert plain in ([0], [1]) and fused == [0]


def test_gloss_lm_prefers_frequent_bigram():
    lm = GlossLM([[0, 1], [0, 1], [0, 2]], n_classes=3)
    assert lm.logp(1, 0) > lm.logp(2, 0)
    assert lm.logp(0, None) > lm.logp(2, None)


def test_nbest_to_lattice_groups_positions():
    lat = nbest_to_lattice([[0, 1], [0, 2], [3, 4, 5]])
    assert lat == [[0], [1, 2]]
    assert nbest_to_lattice([]) == []
