"""Глоссы РЖЯ → связная русская фраза через LLM (HF Inference Providers).

Few-shot: в промпт идут пары из обучающего корпуса с наибольшим пересечением глосс.
Режим --eval меряет chrF на тестовой части корпуса (эталон — человеческий перевод).
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import get_token, hf_hub_download

REPO = 'Arseniy-Polyakov/parallel_corpus_russian_rsl_glosses'
ROUTER = 'https://router.huggingface.co/v1/chat/completions'
DEFAULT_MODEL = 'google/gemma-4-31B-it'
SYSTEM = ('Ты переводчик с русского жестового языка (РЖЯ). На вход — глоссы: жесты, записанные '
          'начальными формами слов в порядке показа. Восстанови одну естественную русскую фразу '
          'с правильными падежами, временем и предлогами. Не добавляй фактов, которых нет в '
          'глоссах. Ответ — только фраза.')
RULES = (' Учти устройство РЖЯ: порядок обычно «тема — рема», служебных слов, связки и '
         'предлогов в глоссах нет, время и число могут не указываться. Восстанавливай '
         'только то, что следует из глоссов и контекста; не добавляй новых людей, '
         'предметов, чисел и причин. Если глоссы обрывочны или противоречат друг другу, '
         'выбери самое вероятное бытовое прочтение, а не переспрашивай. Вопрос и '
         'отрицание могут стоять в конце — сохрани их. Ответ — одна фраза, без кавычек '
         'и пояснений.')
SYSTEM_RULES = SYSTEM + RULES
LATTICE = (' Распознаватель жестов ошибается, поэтому на месте каждого жеста в квадратных скобках '
           'даны варианты через «|», первый — самый вероятный. Выбери по одному варианту на место '
           'так, чтобы вышла осмысленная бытовая фраза.')
PROBS = (' Рядом с вариантом — уверенность распознавателя. Маловероятный вариант бери, только если '
         'он явно лучше вписывается в смысл; если на месте ни один вариант не подходит, пропусти его.')
NBEST = (' Распознаватель жестов выдал несколько вариантов одной фразы, от самого вероятного. '
         'Выбери самый правдоподобный вариант, при необходимости поправь в нём один-два жеста '
         'так, чтобы получилась осмысленная бытовая фраза, и переведи.')
SELECT = ('Распознаватель жестов выдал несколько вариантов одной фразы, от самого вероятного. '
          'Выбери вариант, который лучше всего читается как осмысленная бытовая фраза на русском. '
          'Ответ — только номер варианта.')
COT = (' Сначала выбери самый правдоподобный вариант, затем переведи его. Ответ одной строкой '
       'в формате: «Вариант: <номер>. Перевод: <фраза>».')
REPAIR = (' Распознаватель мог перепутать отдельные жесты: если слово не вписывается в бытовую '
          'фразу, замени его подходящим по смыслу. Главное — понятная фраза, а не дословная цепочка.')


def chain_list(chains, k=8):
    """Цепочки глоссов → нумерованный список для запроса переводчику."""
    return '\n'.join(f'{i}. ' + ' '.join(c) for i, c in enumerate(chains[:k], 1))


JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B-Instruct-2507'
JUDGE = ('Сравни перевод с эталоном. Ответь одной цифрой: 2 — смысл передан (слова могут отличаться), '
         '1 — суть угадывается, но есть искажения, 0 — смысл не передан или искажён.')


def load_corpus(split, extended=False):
    """Загрузить корпус глоссов.
    
    Args:
        split: 'train' или 'test'
        extended: если True, загрузить расширенную версию с словами VOGinfo
    """
    if extended and split == 'train':
        extended_path = Path(__file__).resolve().parent.parent / 'datasets' / 'corpus_train_extended.json'
        if extended_path.exists():
            with open(extended_path, encoding='utf-8') as f:
                return json.load(f)
    with open(hf_hub_download(REPO, f'corpus_{split}.json', repo_type='dataset'), encoding='utf-8') as f:
        return json.load(f)


class Translator:
    def __init__(self, model=DEFAULT_MODEL, shots=12):
        self.model, self.shots = model, shots
        self.train = load_corpus('train')
        self.train_sets = [set(p['rsl'].lower().split()) for p in self.train]
        self.token = get_token()
        self.cost = 0.0

    def examples(self, glosses):
        q = set(glosses.lower().split())
        ranked = sorted(range(len(self.train)),
                        key=lambda i: -len(q & self.train_sets[i]) / (len(q | self.train_sets[i]) or 1))
        return [self.train[i] for i in ranked[:self.shots]][::-1]

    def lattice(self, cands, probs=None):
        """cands: на каждое место список вариантов глоссы, от самого вероятного; probs — их вероятности."""
        if probs is None:
            query = ' '.join('[' + ' | '.join(c) + ']' for c in cands)
            return self(' '.join(w for c in cands for w in c), SYSTEM + LATTICE, query)
        query = ' '.join('[' + ' | '.join(f'{w} {100 * q:.0f}%' for w, q in zip(c, ps)) + ']'
                         for c, ps in zip(cands, probs))
        return self(' '.join(w for c in cands for w in c), SYSTEM + LATTICE + PROBS, query)

    def judge(self, hyp, ref):
        """0/1/2 — передан ли смысл эталона; судит другая модель, не переводчик."""
        msg = [{'role': 'system', 'content': JUDGE},
               {'role': 'user', 'content': f'Эталон: {ref}\nПеревод: {hyp}'}]
        m = re.search(r'[012]', self.chat(msg, 8, JUDGE_MODEL))
        return int(m.group()) if m else None

    def __call__(self, glosses, system=SYSTEM, query=None):
        messages = [{'role': 'system', 'content': system}]
        for ex in self.examples(glosses):
            messages += [{'role': 'user', 'content': ex['rsl']},
                         {'role': 'assistant', 'content': ex['russian']}]
        messages.append({'role': 'user', 'content': query or glosses})
        return self.chat(messages, 80)

    def chat(self, messages, max_tokens, model=None):
        body = json.dumps({'model': model or self.model, 'messages': messages,
                           'max_tokens': max_tokens, 'temperature': 0}).encode()
        req = urllib.request.Request(ROUTER, body, {'Authorization': f'Bearer {self.token}',
                                                    'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.load(r)
        self.cost += (resp.get('usage') or {}).get('estimated_cost') or 0
        return resp['choices'][0]['message']['content'].strip().split('\n')[0]


def chrf(hyp, ref, n=6, beta=2):
    """chrF (Popović 2015): F-мера по символьным n-граммам, пробелы не учитываются."""
    hyp, ref = hyp.lower().replace(' ', ''), ref.lower().replace(' ', '')
    precs, recs = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        m = sum((h & r).values())
        precs.append(m / sum(h.values()))
        recs.append(m / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def evaluate(model, limit, seed, out):
    tr = Translator(model)
    test = load_corpus('test')
    random.Random(seed).shuffle(test)
    test = test[:limit]
    t0 = time.time()

    def one(p):
        try:
            return tr(p['rsl'])
        except Exception as e:  # один вызов без ретраев: ошибка идёт в отчёт, не маскируется
            return f'<ошибка: {e}>'

    with ThreadPoolExecutor(8) as ex:
        hyps = list(ex.map(one, test))
    rows = [{'rsl': p['rsl'], 'ref': p['russian'], 'hyp': h, 'chrf': round(chrf(h, p['russian']), 1)}
            for p, h in zip(test, hyps)]
    copy = sum(chrf(p['rsl'], p['russian']) for p in test) / len(test)
    score = sum(r['chrf'] for r in rows) / len(rows)
    errors = sum(r['hyp'].startswith('<ошибка') for r in rows)
    print(f'{model}: chrF {score:.1f} (просто глоссы как есть: {copy:.1f}), '
          f'{len(rows)} фраз, ошибок {errors}, {time.time() - t0:.0f} с, ${tr.cost:.4f}')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({'model': model, 'chrf': score, 'copy_baseline': copy,
                                     'cost_usd': tr.cost, 'rows': rows}, ensure_ascii=False, indent=1),
                         encoding='utf-8')
    for r in rows[:8]:
        print(f"  {r['rsl']}\n    → {r['hyp']}\n    эталон: {r['ref']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glosses', nargs='*')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--eval', type=int, metavar='N', help='оценить на N тестовых фразах')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='reports/evidence/gloss2text_eval.json')
    a = ap.parse_args()
    if a.eval:
        evaluate(a.model, a.eval, a.seed, a.out)
    else:
        print(Translator(a.model)(' '.join(a.glosses)))


if __name__ == '__main__':
    assert chrf('мама мыла раму', 'мама мыла раму') == 100.0
    assert chrf('абв', 'где') == 0.0
    sys.exit(main())
