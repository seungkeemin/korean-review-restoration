"""Frequency tables learned from (obfuscated, original) review pairs."""

import math
from collections import Counter, defaultdict

from .align import align
from .jamo import to_jamo_list

START = (None, '', '')


def _sorted_map(counter):
    table = defaultdict(list)
    for (s, t), freq in counter.items():
        table[s].append((t, freq))
    for s in table:
        table[s].sort(key=lambda x: x[1], reverse=True)
    return table


def build_single_jamo_map(pairs):
    """syllable-jamo substitution counts: {src_syllable: [(tgt_syllable, freq), ...]} sorted by freq."""
    counter = Counter()
    for src, tgt in pairs:
        for s, t in align(to_jamo_list(src), to_jamo_list(tgt)):
            if s is not None and t is not None:
                counter[(s, t)] += 1
    return _sorted_map(counter)


def build_bigram_map(pairs):
    """Adjacent-pair substitution counts per word: {(s_prev, s): [((t_prev, t), freq), ...]}.

    Words are paired by position after splitting on spaces (spacing is preserved by
    the obfuscation). A START token lets the first syllable form a pair.
    """
    counter = Counter()
    for src, tgt in pairs:
        for sw, tw in zip(src.split(), tgt.split()):
            al = align([START] + to_jamo_list(sw), [START] + to_jamo_list(tw))
            for (s1, t1), (s2, t2) in zip(al, al[1:]):
                if None in (s1, s2, t1, t2):
                    continue
                counter[((s1, s2), (t1, t2))] += 1
    return _sorted_map(counter)


def build_word_freq(texts):
    freq = defaultdict(int)
    for text in texts:
        for w in text.split():
            freq[w] += 1
    return freq


def build_word_bigram_logprob(texts, k_smoothing=1):
    """log P(w2 | w1) with add-k smoothing over the vocabulary, '<S>' marks sentence start."""
    pair_counts, word_counts = Counter(), defaultdict(int)
    for text in texts:
        tokens = text.split()
        if not tokens:
            continue
        tokens = ['<S>'] + tokens
        for w1, w2 in zip(tokens, tokens[1:]):
            pair_counts[(w1, w2)] += 1
            word_counts[w1] += 1
    vocab = len(word_counts)
    return {(w1, w2): math.log((c + k_smoothing) / (word_counts[w1] + vocab * k_smoothing))
            for (w1, w2), c in pair_counts.items()}
