import json
import sys

import numpy as np
import pytest
import torch

from tools import eval_new_classes as evaluation
from tools import finetune_youtube as ft


def test_new_class_indices_do_not_collide_with_normalized_aliases(tmp_path):
    path = tmp_path / 'clips.npz'
    np.savez(path, words=['банк', 'магазин'], lens=[4, 4],
             frames=np.zeros((8, 59, 3), dtype=np.float32))
    mapping = {'веселый': 1, 'дом': 2}
    seqs, labels = ft.load_youtube(path, mapping, add_new=True)
    assert len(seqs) == 2
    assert labels.tolist() == [3, 4]
    assert mapping['дом'] == 2


def test_short_clips_do_not_allocate_classes(tmp_path):
    path = tmp_path / 'clips.npz'
    np.savez(path, words=['банк', 'магазин'], offsets=[0, 3, 7],
             frames=np.zeros((7, 59, 3), dtype=np.float32))
    mapping = {'дом': 0}
    _, labels = ft.load_youtube(path, mapping, add_new=True)
    assert 'банк' not in mapping
    assert labels.tolist() == [1]


def test_same_video_does_not_merge_distinct_words():
    data = {'words': np.array(['страна', 'рана', 'виноград', 'иноград']),
            'video': np.array(['a', 'a', 'b', 'b']), 'variants': np.ones(4)}
    assert ft.file_words(data) == ['страна', 'рана', 'виноград', 'иноград']


def test_ocr_does_not_invent_missing_letters():
    assert ft.norm_ocr('присоединитьс') == 'присоединитьс'


def test_ambiguous_aliases_are_not_assigned_to_arbitrary_class():
    assert ft.class_aliases(['всё', 'все', 'дом/жилище']) == {
        'все': -1, 'дом': 2, 'жилище': 2}


def test_reserved_classifier_indices_are_preserved(tmp_path):
    path = tmp_path / 'clips.npz'
    np.savez(path, words=['банк'], lens=[4], frames=np.zeros((4, 59, 3)))
    mapping = {'все': -1}
    _, labels = ft.load_youtube(path, mapping, add_new=True, first_new_index=2)
    assert labels.tolist() == [2]


@pytest.mark.parametrize('offsets,words', [([0, 5], ['банк']),
    (np.array([0, 8, 4], dtype=np.uint64), ['банк', 'дом'])])
def test_invalid_offsets_are_rejected(tmp_path, offsets, words):
    path = tmp_path / 'clips.npz'
    np.savez(path, words=words, offsets=offsets, frames=np.zeros((4, 59, 3)))
    with pytest.raises(ValueError, match='границы'):
        ft.load_samples(path)


def test_integer_split_mask_does_not_leak_train_into_test(tmp_path):
    path = tmp_path / 'slovo.npz'
    np.savez(path, texts=['дом', 'дом'], offsets=[0, 4, 8], users=[1, 2],
             train=np.array([1, 0]), frames=np.zeros((8, 59, 3)))
    _, _, mask, _, _ = ft.load_slovo(path)
    assert mask.dtype == np.bool_
    assert np.where(~mask)[0].tolist() == [1]


def test_evaluation_selects_actual_intersection_and_excludes_slovo_aliases():
    seq = np.zeros((4, 59, 3))
    train = [(word, seq) for word in ['дом', 'жилище', 'банк', 'магазин']]
    test = [(word, seq) for word in ['жилище', 'банк', 'улица']]
    selected = evaluation.select_samples(['дом/жилище', 'банк', 'магазин'],
                                         train, test, ['дом/жилище'])
    assert [word for word, _ in selected] == ['банк']


def test_holdout_checks_provenance_sources_and_sample_overlap(tmp_path):
    train, test, changed = [tmp_path / name for name in ['train.npz', 'test.npz', 'changed.npz']]
    for value, path in enumerate([train, test, changed]):
        np.savez(path, words=['банк'], lens=[4], frames=np.full((4, 59, 3), value))
    samples = ft.load_samples(test)
    with pytest.raises(ValueError, match='истории обучения'):
        evaluation.verify_holdout({}, [train], test, samples)
    unverified = evaluation.verify_holdout({}, [train], test, samples, True)
    assert not unverified['finetune_holdout_verified']
    incomplete = {'training': {'sources': [{'sha256': ft.file_sha256(train)}]}}
    with pytest.raises(ValueError, match='истории обучения'):
        evaluation.verify_holdout(incomplete, [train], test, samples)
    assert not evaluation.verify_holdout(incomplete, [train], test, samples, True)[
        'finetune_holdout_verified']
    checkpoint = {'training': {'schema_version': 1,
                               'sources': [{'sha256': ft.file_sha256(train)}],
                               'sample_sha256': [ft.sample_sha256(ft.load_samples(train)[0][1])]}}
    result = evaluation.verify_holdout(checkpoint, [train], test, samples)
    assert result['finetune_holdout_verified']
    assert not result['base_provenance_verified']
    with pytest.raises(ValueError, match='совпадает с обучающим'):
        evaluation.verify_holdout(checkpoint, [train], train, ft.load_samples(train))
    with pytest.raises(ValueError, match='manifest'):
        evaluation.verify_holdout(checkpoint, [changed], test, samples)
    checkpoint['training']['sample_sha256'] = [ft.sample_sha256(samples[0][1])]
    with pytest.raises(ValueError, match='образцы из обучения'):
        evaluation.verify_holdout(checkpoint, [train], test, samples)


def test_fixed_epoch_training_saves_evaluated_model_and_provenance(tmp_path, monkeypatch):
    class TinySignNet(torch.nn.Module):
        def __init__(self, n_classes, feat):
            super().__init__()
            self.head = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(),
                                            torch.nn.Linear(feat, n_classes))

        def forward(self, x):
            return self.head(x.mean(1))

    monkeypatch.setattr(ft, 'SignNet', TinySignNet)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    slovo, youtube, base, out = [tmp_path / name for name in
                                ['slovo.npz', 'yt.npz', 'base.pt', 'out.pt']]
    names = ['а', 'б', 'в', 'г', 'д']
    rng = np.random.default_rng(7)
    np.savez(slovo, texts=np.repeat(names, 2), train=np.tile([True, False], 5),
             users=np.arange(10), offsets=np.arange(11) * 4,
             frames=rng.uniform(0.2, 0.8, (40, 59, 3)).astype(np.float32))
    np.savez(youtube, words=['банк'] * 3 + ['магазин'] * 3, lens=[4] * 3 + [3] * 3,
             frames=rng.uniform(0.2, 0.8, (21, 59, 3)).astype(np.float32))
    torch.save({'model': TinySignNet(5, ft.HFEAT).state_dict(),
                'names': names[::-1], 'kind': 'holistic'}, base)
    calls = []
    original_topk = ft.topk

    def record_topk(logits, y, k):
        calls.append(np.asarray(y).tolist())
        return original_topk(logits, y, k)

    monkeypatch.setattr(ft, 'topk', record_topk)
    monkeypatch.setattr(sys, 'argv', ['finetune', '--slovo', str(slovo), '--youtube', str(youtube),
                                    '--checkpoint', str(base), '--out', str(out), '--epochs', '5',
                                    '--bs', '2', '--add-new', '--yt-weight', '1'])
    ft.main()
    saved = torch.load(out, weights_only=False)
    result = json.loads(out.with_suffix('.eval.json').read_text())
    assert calls[0] == [4, 3, 2, 1, 0]
    assert len(calls) == 6
    assert saved['names'] == names[::-1] + ['банк']
    assert saved['model']['head.2.weight'].shape[0] == 6
    assert saved['training']['selection'] == 'fixed_epochs'
    assert saved['training']['sources'][0]['sha256'] == ft.file_sha256(youtube)
    assert not saved['training']['base_provenance_verified']
    assert result['checkpoint_sha256'] == ft.file_sha256(out)
    assert result['yt_samples'] == 3
    assert 'yt_top1' not in result and 'yt_train_top1' in result
    model = TinySignNet(6, ft.HFEAT)
    model.load_state_dict(saved['model'])
    seqs, y, train_mask, _, data_names = ft.load_slovo(slovo)
    expected_y = np.array([saved['names'].index(data_names[label]) for label in y[~train_mask]])
    logits = ft.predict(model, [seq for seq, keep in zip(seqs, ~train_mask) if keep], 'cpu')
    assert original_topk(logits, expected_y, 1) == result['final_top1']
    with pytest.raises(SystemExit):
        ft.main()
    empty_out = tmp_path / 'no_updates.pt'
    sys.argv[sys.argv.index('--out') + 1] = str(empty_out)
    sys.argv[sys.argv.index('--bs') + 1] = '999'
    with pytest.raises(SystemExit):
        ft.main()
    assert not empty_out.exists()
