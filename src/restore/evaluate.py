"""Word and sentence accuracy, compared position by position (spacing is preserved)."""


def accuracy(predictions, targets):
    words = correct_words = correct_sent = 0
    for p, t in zip(predictions, targets):
        pw, tw = p.split(), t.split()
        n = min(len(pw), len(tw))
        correct_words += sum(a == b for a, b in zip(pw[:n], tw[:n]))
        words += n
        correct_sent += p == t
    return correct_words / words, correct_sent / len(targets)
