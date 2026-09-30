"""Beam search per word, then Viterbi over the sentence with word-bigram scores."""

import math
from dataclasses import dataclass, field

from .jamo import from_jamo_list, to_jamo_list
from .maps import (START, build_bigram_map, build_single_jamo_map, build_word_bigram_logprob,
                   build_word_freq)


@dataclass
class Model:
    single_map: dict
    bigram_map: dict
    word_freq: dict
    word_bigram: dict
    backward: bool = False
    params: dict = field(default_factory=dict)


def fit(pairs, backward=True):
    """Learn every table from (obfuscated, original) pairs. backward=True reverses the strings first."""
    pairs = [(s[::-1], t[::-1]) if backward else (s, t) for s, t in pairs]
    outputs = [t for _, t in pairs]
    return Model(build_single_jamo_map(pairs), build_bigram_map(pairs), build_word_freq(outputs),
                 build_word_bigram_logprob(outputs), backward)


def word_candidates(src_word, model, lam=0.1, topk=160, beam_size=160, cand_k=15):
    """Top cand_k restorations of one word as [(word, score)].

    Each step extends the beam with (a) bigram substitutions whose previous target
    syllable matches the beam's last one, scored log(1 + f_bigram) + lam log(1 + f_single),
    and (b) single substitutions scored lam log(1 + f_single). The final score adds
    log(1 + word frequency) so real words win over near-misses.
    """
    s = [START] + to_jamo_list(src_word)
    beam = [([START], START, 0.0)]
    for i in range(1, len(s)):
        bi = model.bigram_map.get((s[i - 1], s[i]), [])[:topk]
        singles = model.single_map.get(s[i], [])
        single_freq = dict(singles)
        new_beam = []
        for decoded, last, sc in beam:
            for (t_prev, t_i), f_bi in bi:
                if t_prev == last:
                    score = sc + math.log(1 + f_bi) + lam * math.log(1 + single_freq.get(t_i, 0))
                    new_beam.append((decoded + [t_i], t_i, score))
            for t_i, f_s in singles[:topk]:
                new_beam.append((decoded + [t_i], t_i, sc + lam * math.log(1 + f_s)))
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_size]
        if not beam:
            break

    cands = {}
    for decoded, _, sc in beam:
        w = from_jamo_list([j for j in decoded if j[0] is not None])
        score = sc + math.log(1 + model.word_freq.get(w, 0))
        if score > cands.get(w, -math.inf):
            cands[w] = score
    ranked = sorted(cands.items(), key=lambda x: x[1], reverse=True)
    return ranked[:cand_k] if ranked else [(src_word, -100.0)]


def viterbi(layers, word_bigram, context_weight=2.5, start_penalty=-15.0, unseen_penalty=-12.0):
    """Best path through candidate layers: sum of candidate scores + context_weight * log P(w_t | w_{t-1})."""
    dp = [{i: (sc + word_bigram.get(('<S>', w), start_penalty) * context_weight, -1)
           for i, (w, sc) in enumerate(layers[0])}]
    for t in range(1, len(layers)):
        row = {}
        for ci, (cw, cs) in enumerate(layers[t]):
            best, arg = -math.inf, -1
            for pi, (pw, _) in enumerate(layers[t - 1]):
                total = dp[t - 1][pi][0] + cs + word_bigram.get((pw, cw), unseen_penalty) * context_weight
                if total > best:
                    best, arg = total, pi
            row[ci] = (best, arg)
        dp.append(row)
    idx = max(dp[-1], key=lambda i: dp[-1][i][0])
    words = []
    for t in range(len(layers) - 1, -1, -1):
        words.append(layers[t][idx][0])
        idx = dp[t][idx][1]
    return words[::-1]


def restore(sentence, model, lam=0.1, topk=160, beam_size=160, cand_k=15, context_weight=2.5):
    """Restore one obfuscated review. Parameters default to the final submission settings."""
    text = sentence[::-1] if model.backward else sentence
    tokens = text.split()
    if not tokens:
        return sentence
    layers = [word_candidates(tok, model, lam, topk, beam_size, cand_k) for tok in tokens]
    out = " ".join(viterbi(layers, model.word_bigram, context_weight))
    return out[::-1] if model.backward else out
