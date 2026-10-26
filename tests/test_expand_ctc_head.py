import torch

from tools.expand_ctc_head import expand_checkpoint, expand_rows, extend_names


def test_extend_names_dedups_synonyms_and_yo():
    names = ['мать', 'там/туда', 'ёлка']
    new = extend_names(names, ['Мать', 'ТУДА', 'елка', 'бобр', 'Бобр', ''])
    assert new == names + ['бобр']


def test_expand_rows_keeps_old_rows():
    w = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    out = expand_rows(w, 2)
    assert out.shape == (6, 2)
    assert torch.equal(out[:4], w)
    assert float(out[4:].abs().sum()) == 0.0


def test_expand_checkpoint_keeps_encoder_and_old_head(tmp_path):
    names = ['мать', 'там/туда']
    state = {'inp.0.weight': torch.ones(3, 2), 'head.weight': torch.ones(3, 2),
             'head.bias': torch.zeros(3), 'aux.weight': torch.ones(2, 2),
             'aux.bias': torch.zeros(2)}
    ck = {'names': names, 'model': state, 'aux': 2, 'feat': 496,
          'arch': {'d': 2, 'layers': 1, 'heads': 1, 'dropout': 0.0}}
    out = expand_checkpoint(ck, ['бобр', 'авария'])
    assert out['names'] == names + ['авария', 'бобр']
    assert out['expanded_added'] == 2 and out['aux'] == 4
    assert out['model']['head.weight'].shape == (5, 2)
    assert torch.equal(out['model']['head.weight'][:3], torch.ones(3, 2))
    assert float(out['model']['head.weight'][3:].abs().sum()) == 0.0
    assert torch.equal(out['model']['inp.0.weight'], torch.ones(3, 2))
    path = tmp_path / 'head.pt'
    torch.save(out, path)
    assert torch.load(path, weights_only=False)['names'][-1] == 'бобр'
