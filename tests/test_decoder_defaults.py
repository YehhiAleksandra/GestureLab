"""Дефолты боевого декодера: один источник правды и зафиксированный по смыслу конфиг (B67)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dialogue_translator import TUNED_DECODER  # noqa: E402


def test_tuned_decoder_is_the_candidate_frozen_by_meaning():
    # заморозан по канонизированному F1 на tune (этап 32) и подтверждён по переводу (этап 33)
    assert TUNED_DECODER == {'decoder': 'beam', 'beam': 24, 'lm_weight': 0.0,
                             'word_bonus': 4.0, 'smooth': 3, 'nbest': 20}


def test_no_second_copy_of_the_pair():
    import check_translation
    import eval_phrase_clips
    assert check_translation.TUNED_DECODER is TUNED_DECODER, 'проверка перевода со своим декодером'
    assert eval_phrase_clips.TUNED_DECODER is TUNED_DECODER, 'оценка со своим декодером'
