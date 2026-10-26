"""Сколько слов тестовой речи встречается в речи учебных видео (потолок для разметки жестов по голосу)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymorphy3

morph = pymorphy3.MorphAnalyzer()
STOP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO"}


def lemmas(path: Path) -> list[str]:
    text = " ".join(s["text"] for s in json.loads(path.read_text(encoding="utf-8")))
    out = []
    for w in re.findall(r"[а-яё]+", text.lower()):
        p = morph.parse(w)[0]
        if p.tag.POS not in STOP_POS:
            out.append(p.normal_form)
    return out


def main(test: str, ref_dir: str = "reports/ref", min_count: int = 3) -> None:
    test_path = Path(test)
    train = Counter()
    for f in Path(ref_dir).glob("*.json"):
        if f.resolve() != test_path.resolve():
            train.update(lemmas(f))
    test_l = lemmas(test_path)
    for k in (1, min_count, 10):
        hit = [w for w in test_l if train[w] >= k]
        print(f"слово встречается в учебной речи >= {k} раз: {len(hit)}/{len(test_l)} = {len(hit) / len(test_l):.1%} "
              f"токенов, {len(set(hit))}/{len(set(test_l))} лемм")
    print(f"учебная речь: {sum(train.values())} значимых слов, {len(train)} лемм")
    tc = Counter(test_l)
    print("частые в тесте:", ", ".join(f"{w} {n}/{train[w]}" for w, n in tc.most_common(40)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/ref/holy_sepulchre_reference.json")
