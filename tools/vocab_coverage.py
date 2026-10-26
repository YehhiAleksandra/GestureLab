"""Доля слов эталонной речи, которые вообще есть среди 1000 классов Slovo (потолок для любой модели на Slovo)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymorphy3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slovo_repo"))
from constants import classes  # noqa: E402

morph = pymorphy3.MorphAnalyzer()
lemma = lambda w: morph.parse(w)[0].normal_form  # noqa: E731
STOP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO"}


def main(ref_path: str) -> None:
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    words = re.findall(r"[а-яё]+", " ".join(s["text"] for s in ref).lower())
    content = [lemma(w) for w in words if morph.parse(w)[0].tag.POS not in STOP_POS]
    vocab = {lemma(c.lower()) for c in classes.values() if len(c) > 1}
    hit = [w for w in content if w in vocab]
    print(f"слов в речи: {len(words)}, значимых: {len(content)}, уникальных лемм: {len(set(content))}")
    print(f"покрытие словарём Slovo: {len(hit)}/{len(content)} = {len(hit) / len(content):.1%} токенов, "
          f"{len(set(hit))}/{len(set(content))} = {len(set(hit)) / len(set(content)):.1%} лемм")
    print("совпавшие:", ", ".join(f"{w}×{n}" for w, n in Counter(hit).most_common(40)))
    miss = Counter(w for w in content if w not in vocab)
    print("частые без жеста в Slovo:", ", ".join(f"{w}×{n}" for w, n in miss.most_common(40)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/ref/holy_sepulchre_reference.json")
