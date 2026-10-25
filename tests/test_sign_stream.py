import numpy as np
import pytest

torch = pytest.importorskip('torch')

from sign_stream import BLANK, CTCSignNet, greedy_decode, out_lengths


def test_out_lengths_halves_time_ceil():
    assert out_lengths([1, 2, 3, 4, 101]).tolist() == [1, 1, 2, 2, 51]


def test_greedy_decode_collapses_repeats_and_drops_blank():
    logits = torch.full((9, 4), -10.0)
    seq = [BLANK, BLANK, 1, 1, BLANK, 3, 3, 3, 1]
    for t, c in enumerate(seq):
        logits[t, c] = 10.0
    ids, conf = greedy_decode(logits)
    assert ids == [0, 2, 0]  # классы сдвинуты: 1 → 0, 3 → 2
    assert len(conf) == 3 and all(0.9 < c <= 1.0 for c in conf)


def test_greedy_decode_all_blank_gives_empty():
    ids, conf = greedy_decode(torch.zeros((5, 3)))
    assert ids == [] and conf == []


def test_ctc_model_shape_and_masking():
    model = CTCSignNet(7, d=32, layers=1, heads=2, dropout=0.0)
    x = torch.randn(2, 40, 248)
    out = model(x)
    assert out.shape == (2, 20, 8)

    padded = model(x, torch.tensor([40, 21]))
    assert padded.shape == (2, 20, 8) and torch.isfinite(padded).all()


def test_ctc_targets_index_logits_directly():
    """Регресс: target 1 = логит 1, а не 2; target 0 был бы blank и терялся."""
    logits = torch.full((3, 2), -30.0)
    logits[:, 0] = 30.0
    logits[1, 0], logits[1, 1] = -30.0, 30.0
    loss = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)(
        logits[None].log_softmax(-1).transpose(0, 1), torch.tensor([1]),
        torch.tensor([3]), torch.tensor([1]))
    assert loss.item() < 1e-3


def test_ctc_model_is_trainable():
    model = CTCSignNet(3, d=16, layers=1, heads=2, dropout=0.0)
    logits = model(torch.randn(1, 30, 248))
    # цели CTCLoss — индексы логитов напрямую, 0 занят blank: класс c → c + 1
    loss = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)(
        logits.log_softmax(-1).transpose(0, 1), torch.tensor([1, 3]),
        torch.tensor([15]), torch.tensor([2]))
    loss.backward()
    assert np.isfinite(loss.item()) and loss.item() >= 0
    assert model.head.weight.grad is not None


def test_ctc_aux_head_pools_only_valid_steps():
    model = CTCSignNet(7, d=32, layers=1, heads=2, dropout=0.0, aux=7)
    x = torch.randn(3, 41, 248)
    logits, aux = model(x, torch.tensor([41, 20, 1]), aux=True)
    assert logits.shape[:2] == (3, 21) and aux.shape == (3, 7)
    assert torch.isfinite(aux).all()
