"""Оценка на отдельном источнике с проверкой происхождения данных дообучения."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slovo_model import KINDS, SignNet, T, resample
from tools.finetune_youtube import (
    class_aliases,
    file_sha256,
    load_samples,
    sample_sha256,
)


def verify_holdout(checkpoint, train_paths, test_path, samples, allow_unverified=False):
    training = checkpoint.get('training')
    complete = bool(training and training.get('schema_version') == 1
                    and isinstance(training.get('sample_sha256'), list)
                    and training['sample_sha256'] and isinstance(training.get('sources'), list))
    if not complete and not allow_unverified:
        raise ValueError('Нет полной истории обучения checkpoint; независимость теста не подтверждена')
    train_hashes = {file_sha256(path) for path in train_paths}
    if file_sha256(test_path) in train_hashes:
        raise ValueError('Источник теста совпадает с обучающим')
    if training and train_hashes != {record['sha256'] for record in training['sources']}:
        raise ValueError('Источники обучения не совпадают с сохранённым manifest')
    seen = set(training.get('sample_sha256', [])) if training else set()
    seen.update(sample_sha256(seq) for path in train_paths for _, seq in load_samples(path))
    overlap = sum(sample_sha256(seq) in seen for _, seq in samples)
    if overlap:
        raise ValueError(f'В тесте есть образцы из обучения: {overlap}')
    return {
        'finetune_holdout_verified': complete,
        'base_provenance_verified': bool(complete and training.get('base_provenance_verified')),
        'exact_overlap_samples': overlap,
        'limitation': 'Хеши исключают точные копии, но не другие обрезки одного видео; '
                      'независимость исполнителей и происхождение базовой модели не доказаны.',
    }


def select_samples(names, train_samples, test_samples, slovo_names, scope='new'):
    mapping = class_aliases(names)
    slovo = set(class_aliases(slovo_names))
    seen = {word for word, _ in train_samples}
    eligible = {word for word in seen if mapping.get(word, -1) >= 0
                and (scope == 'all' or (word in slovo) == (scope == 'known'))}
    return [(word, seq) for word, seq in test_samples if word in eligible]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-source', nargs='+', required=True)
    ap.add_argument('--test-source', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--slovo', default='datasets/slovo/slovo_holistic.npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--scope', choices=('new', 'known', 'all'), default='new')
    ap.add_argument('--allow-unverified', action='store_true',
                    help='диагностика старых checkpoint без утверждения о независимости')
    a = ap.parse_args()
    if Path(a.out).exists():
        ap.error('Выберите новый --out: результат уже существует')
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    names = checkpoint['names']
    with np.load(a.slovo) as data:
        slovo_names = sorted({str(word) for word in data['texts']})
    training = checkpoint.get('training')
    if training and file_sha256(a.slovo) != training['slovo']['sha256']:
        ap.error('Slovo не совпадает с manifest обучения')
    train_samples = [sample for path in a.train_source for sample in load_samples(path)]
    test_samples = load_samples(a.test_source)
    samples = select_samples(names, train_samples, test_samples, slovo_names, a.scope)
    if not samples:
        ap.error('Нет подходящих образцов для выбранного словаря')
    try:
        provenance = verify_holdout(checkpoint, a.train_source, a.test_source, samples,
                                    a.allow_unverified)
    except ValueError as error:
        ap.error(str(error))
    mapping = class_aliases(names)
    feature_fn, feat = KINDS[checkpoint.get('kind', 'hands')]
    model = SignNet(len(names), feat=feat)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    per_sample = []
    with torch.no_grad():
        for word, seq in samples:
            x = feature_fn(resample(seq, T))[None]
            logits = model(torch.from_numpy(x).to(dev))[0].cpu()
            order = logits.argsort(descending=True)[:3].tolist()
            per_sample.append({'word': word, 'sample_sha256': sample_sha256(seq),
                               'top1': order[0] == mapping[word],
                               'top3': mapping[word] in order, 'pred': names[order[0]]})
    cohort = sorted((item['word'], item['sample_sha256']) for item in per_sample)
    result = {
        'checkpoint': Path(a.checkpoint).name,
        'checkpoint_sha256': file_sha256(a.checkpoint),
        'classes': len(names),
        'scope': a.scope,
        'train_sources': [Path(path).name for path in a.train_source],
        'test_source': Path(a.test_source).name,
        'test_source_sha256': file_sha256(a.test_source),
        'evaluated_words': sorted({word for word, _ in samples}),
        'evaluated_word_count': len({word for word, _ in samples}),
        'test_samples': len(samples),
        'source_samples': len(test_samples),
        'cohort_sha256': hashlib.sha256(json.dumps(cohort, ensure_ascii=False).encode()).hexdigest(),
        'top1': sum(item['top1'] for item in per_sample) / len(samples),
        'top3': sum(item['top3'] for item in per_sample) / len(samples),
        'provenance': provenance,
        'per_sample': per_sample,
    }
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('per_sample', 'evaluated_words')}, ensure_ascii=False, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
