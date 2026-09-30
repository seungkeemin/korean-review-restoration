"""Levenshtein alignment of two token sequences."""


def align(src, tgt):
    """Minimum-edit alignment as a list of (src_token | None, tgt_token | None) pairs.

    Deletions come out as (s, None), insertions as (None, t), matches and
    substitutions as (s, t). Ties prefer deletion, then insertion, then diagonal,
    the same order as the original notebook, so learned counts do not change.
    """
    n, m = len(src), len(tgt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if src[i - 1] == tgt[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((src[i - 1], None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            pairs.append((None, tgt[j - 1]))
            j -= 1
        else:
            pairs.append((src[i - 1], tgt[j - 1]))
            i -= 1
            j -= 1
    return pairs[::-1]
