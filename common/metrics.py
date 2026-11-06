"""The numbers every branch reports, computed one way.

    report(labels, probs)   # labels: 1 = advance, 0 = retract; probs: P(advance)

For a table that compares models, pool the predictions of all five folds' test groups
(416 cases) and call this once; a single fold's test group is only 69-123 cases.
"""


def report(labels, probs, threshold=0.5):
    labels = [int(v) for v in labels]
    probs = [float(v) for v in probs]
    predictions = [int(p >= threshold) for p in probs]
    advance = [p for p, l in zip(predictions, labels) if l == 1]
    retract = [p for p, l in zip(predictions, labels) if l == 0]
    # A group with one class only (possible on a small fold) has no recall for the other.
    advance_accuracy = sum(advance) / len(advance) if advance else None
    retract_accuracy = 1 - sum(retract) / len(retract) if retract else None
    both = advance and retract
    return {"n": len(labels),
            "overall_accuracy": sum(p == l for p, l in zip(predictions, labels)) / len(labels),
            "advance_accuracy": advance_accuracy,
            "retract_accuracy": retract_accuracy,
            "balanced_accuracy": (advance_accuracy + retract_accuracy) / 2 if both else None,
            "auroc": auroc(labels, probs) if both else None,
            "always_advance_accuracy": len(advance) / len(labels)}


def auroc(labels, probs):
    """Chance that a random advance case scores above a random retract case (ties half)."""
    ranked = sorted(range(len(probs)), key=lambda i: probs[i])
    rank = [0.0] * len(probs)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and probs[ranked[j + 1]] == probs[ranked[i]]:
            j += 1
        for k in range(i, j + 1):
            rank[ranked[k]] = (i + j) / 2 + 1
        i = j + 1
    positives = sum(labels)
    negatives = len(labels) - positives
    rank_sum = sum(r for r, l in zip(rank, labels) if l == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
