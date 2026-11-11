"""chrF score for machine translation evaluation.

Character n-gram F-score as described in:
Popovic (2015) "chrF: character n-gram F-score for automatic MT evaluation"
"""

from __future__ import annotations

from collections import Counter


def compute_chrf(
    hypothesis: str,
    reference: str,
    max_n: int = 6,
    beta: float = 2.0,
) -> float:
    """Compute chrF score.

    Args:
        hypothesis: Candidate translation.
        reference: Reference translation.
        max_n: Maximum character n-gram order.
        beta: Parameter weighting recall vs precision.

    Returns:
        chrF score (0 to 1).
    """
    if not hypothesis or not reference:
        return 0.0

    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for n in range(1, max_n + 1):
        hyp_ngrams = _char_ngrams(hypothesis, n)
        ref_ngrams = _char_ngrams(reference, n)

        if not hyp_ngrams or not ref_ngrams:
            continue

        # Compute overlap
        overlap = 0
        for ngram, hyp_count in hyp_ngrams.items():
            overlap += min(hyp_count, ref_ngrams.get(ngram, 0))

        precision = overlap / sum(hyp_ngrams.values())
        recall = overlap / sum(ref_ngrams.values())

        total_precision += precision
        total_recall += recall
        count += 1

    if count == 0:
        return 0.0

    avg_precision = total_precision / count
    avg_recall = total_recall / count

    if avg_precision + avg_recall == 0:
        return 0.0

    # F-beta score
    beta_sq = beta ** 2
    chrf = (1 + beta_sq) * avg_precision * avg_recall / (
        beta_sq * avg_precision + avg_recall
    )

    return round(chrf, 4)


def corpus_chrf(
    hypotheses: list[str],
    references: list[str],
    max_n: int = 6,
    beta: float = 2.0,
) -> float:
    """Compute corpus-level chrF score.

    Args:
        hypotheses: List of candidate translations.
        references: List of reference translations.
        max_n: Maximum character n-gram order.
        beta: F-score beta parameter.

    Returns:
        Average chrF score across the corpus.
    """
    assert len(hypotheses) == len(references)

    if not hypotheses:
        return 0.0

    scores = [
        compute_chrf(hyp, ref, max_n=max_n, beta=beta)
        for hyp, ref in zip(hypotheses, references)
    ]
    return round(sum(scores) / len(scores), 4)


def _char_ngrams(text: str, n: int) -> Counter:
    """Extract character n-grams, treating spaces as word boundaries."""
    # Include spaces to capture word boundary information
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return Counter(ngrams)
