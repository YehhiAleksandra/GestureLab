"""Какая доля глосс бытового корпуса РЖЯ есть среди жестов Slovo и Easy_sign."""
import csv
import json
import sys
from collections import Counter

from huggingface_hub import hf_hub_download

REPO = 'Arseniy-Polyakov/parallel_corpus_russian_rsl_glosses'


def norm(w):
    return w.lower().replace('ё', 'е').strip().strip('.,!?')


def split_classes(name):
    # Easy_sign пишет синонимы через «/» или «;»: «там/туда», «вредный; озорной»
    return {norm(p) for p in name.replace(';', '/').split('/') if norm(p)}


def load_pairs(split):
    path = hf_hub_download(REPO, f'corpus_{split}.json', repo_type='dataset')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main():
    with open('datasets/slovo/annotations.csv', encoding='utf-8') as f:
        slovo = set().union(*(split_classes(r['text']) for r in csv.DictReader(f, delimiter='\t')))
    with open('models/easy_sign/RSL_class_list.txt', encoding='utf-8') as f:
        easy = set().union(*(split_classes(l.split('\t', 1)[1]) for l in f if '\t' in l))

    for split in ('train', 'test'):
        pairs = load_pairs(split)
        tokens = Counter(norm(w) for p in pairs for w in p['rsl'].split())
        total = sum(tokens.values())
        print(f'\n== {split}: {len(pairs)} фраз, {total} глосс, {len(tokens)} разных')
        for name, vocab in (('Slovo', slovo), ('Easy_sign', easy), ('оба', slovo | easy)):
            cov = sum(c for w, c in tokens.items() if w in vocab)
            full = sum(all(norm(w) in vocab for w in p['rsl'].split()) for p in pairs)
            print(f'  {name:9} {len(vocab):5} классов: токены {100 * cov / total:5.1f}%, '
                  f'фраз целиком {full} ({100 * full / len(pairs):.1f}%)')
        missing = Counter({w: c for w, c in tokens.items() if w not in slovo | easy})
        print('  чаще всего нет жеста:', missing.most_common(20))


if __name__ == '__main__':
    sys.exit(main())
