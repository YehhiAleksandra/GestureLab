import pytest


def test_heldout_phrases_do_not_leak_into_lm(monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('huggingface_hub')
    from tools import eval_lattice

    names = [f'w{i}' for i in range(30)]
    seqs = [' '.join(f'w{j}' for j in range(i, i + 3)) for i in range(0, 30, 3)]
    index = {n: i for i, n in enumerate(names)}
    train = [{'rsl': s} for s in seqs]
    test = [{'rsl': s, 'russian': f'фраза {i}'} for i, s in enumerate(seqs)]
    monkeypatch.setattr(eval_lattice, 'load_corpus',
                        lambda split: train if split == 'train' else test)
    monkeypatch.setattr(eval_lattice, 'corpus_phrases',
                        lambda idx, split: [tuple(idx[w] for w in p['rsl'].split())
                                            for p in train])
    d = {'te_by_class': {i: [0] for i in range(30)}}

    lm_seqs, phrases = eval_lattice.heldout_phrases(names, d, dev_size=0)

    assert phrases
    held = {tuple(p['ids']) for p in phrases}
    assert held.isdisjoint({tuple(s) for s in lm_seqs})
    assert len(held) + len(lm_seqs) == len(seqs)


def test_ensemble_logp_averages_probabilities_not_logits():
    torch = pytest.importorskip('torch')
    from tools.eval_lattice import ensemble_logp

    a = torch.tensor([[0.0, 0.0], [10.0, 0.0]])   # p = [.5, .5], [1, 0]
    b = torch.tensor([[0.0, 0.0], [-10.0, 0.0]])  # p = [.5, .5], [0, 1]
    out = torch.from_numpy(ensemble_logp([a, b])).exp()
    assert torch.allclose(out[0], torch.full((2,), 0.5), atol=1e-6)
    assert torch.allclose(out[1], torch.full((2,), 0.5), atol=1e-6)
