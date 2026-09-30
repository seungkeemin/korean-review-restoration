"""Hold-out accuracy on the Dacon training set (the data itself is not in this repository).

    python scripts/holdout_eval.py path/to/train.csv [--frac 0.1] [--seed 0] [--forward]

Tables are learned on the rest of the training set and the held-out reviews are restored,
so the score is out of sample. Accuracy is word-level and sentence-level exact match.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restore import accuracy, fit, restore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_csv")
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forward", action="store_true", help="disable the backward (reversed) mode")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.train_csv, encoding="utf-8-sig")))
    random.Random(a.seed).shuffle(rows)
    n_test = int(len(rows) * a.frac)
    test, train = rows[:n_test], rows[n_test:]

    t0 = time.time()
    model = fit([(r["input"], r["output"]) for r in train], backward=not a.forward)
    t1 = time.time()
    preds = [restore(r["input"], model) for r in test]
    t2 = time.time()
    word_acc, sent_acc = accuracy(preds, [r["output"] for r in test])
    ident = accuracy([r["input"] for r in test], [r["output"] for r in test])
    print(f"train {len(train)}, held out {len(test)}, seed {a.seed}, mode {'forward' if a.forward else 'backward'}")
    print(f"word accuracy {word_acc:.4f}, sentence accuracy {sent_acc:.4f}")
    print(f"(no restoration: word {ident[0]:.4f}, sentence {ident[1]:.4f})")
    print(f"fit {t1 - t0:.0f}s, restore {t2 - t1:.0f}s")

    vocab = {w for r in train for w in r["output"].split()}
    seen = [0, 0]
    unseen = [0, 0]
    for p, r in zip(preds, test):
        for pw, tw in zip(p.split(), r["output"].split()):
            bucket = seen if tw in vocab else unseen
            bucket[0] += pw == tw
            bucket[1] += 1
    total = seen[1] + unseen[1]
    print(f"held-out words not seen in training outputs: {unseen[1] / total:.4f}")
    print(f"accuracy on seen words {seen[0] / seen[1]:.4f}, on unseen words {unseen[0] / unseen[1]:.4f}")


if __name__ == "__main__":
    main()
