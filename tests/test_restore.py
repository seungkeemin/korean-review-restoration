"""Tests on synthetic text; the Dacon data is not in the repository.

The toy obfuscation below swaps a few initial/medial jamo deterministically and adds a
final consonant to open syllables. Real obfuscation is stochastic, but the pipeline
should at least invert a deterministic rule it has seen in training.
"""

import random

from restore import accuracy, fit, from_jamo_list, restore, to_jamo_list, viterbi
from restore.align import align
from restore.jamo import compose_syllable

CHO = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅈ": "ㅉ", "ㅅ": "ㅆ"}
JUNG = {"ㅏ": "ㅑ", "ㅓ": "ㅕ", "ㅗ": "ㅛ", "ㅜ": "ㅠ"}

WORDS = ["방이", "넓고", "깨끗해요", "직원이", "친절합니다", "위치가", "좋아서", "다시", "오고",
         "싶어요", "조식은", "그냥", "그래요", "주차가", "편했어요", "가격", "대비", "만족"]


def obfuscate(text):
    out = []
    for cho, jung, jong in to_jamo_list(text):
        if jung == "":
            out.append(cho)
            continue
        jong = jong or "ㅁ"
        out.append(compose_syllable(CHO.get(cho, cho), JUNG.get(jung, jung), jong))
    return "".join(out)


def corpus(n, seed):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 6))) for _ in range(n)]


def test_jamo_roundtrip_keeps_non_hangul():
    text = "방이 넓고 깨끗해요!! 10점 만점에 9점 :)"
    assert from_jamo_list(to_jamo_list(text)) == text


def test_alignment_of_equal_sequences_is_all_matches():
    seq = list("abcde")
    assert align(seq, seq) == list(zip(seq, seq))
    assert align(list("abc"), list("abd"))[-1] == ("c", "d")


def test_learned_tables_invert_the_toy_obfuscation():
    train = [(obfuscate(t), t) for t in corpus(300, seed=1)]
    test = corpus(40, seed=2)                      # new word orders, known words
    for backward in (True, False):
        model = fit(train, backward=backward)
        preds = [restore(obfuscate(t), model) for t in test]
        word_acc, sent_acc = accuracy(preds, test)
        assert word_acc == 1.0 and sent_acc == 1.0


def test_viterbi_uses_word_context_to_break_ties():
    layers = [[("가격", 0.0)], [("대비", 0.0), ("대미", 0.0)]]
    bigram = {("<S>", "가격"): -0.1, ("가격", "대비"): -0.1}
    assert viterbi(layers, bigram) == ["가격", "대비"]


def test_accuracy_counts_words_by_position():
    assert accuracy(["가 나 다"], ["가 나 라"]) == (2 / 3, 0.0)
