"""BLEU score implementation for machine translation evaluation.

Implements the BLEU metric as described in:
Papineni et al. (2002) "BLEU: a Method for Automatic Evaluation
of Machine Translation"
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def compute_bleu(
    hypothesis: str,
    reference: str | list[str],
    max_n: int = 4,
    smoothing: bool = True,
) -> dict[str, float]:
    """Compute BLEU score with full breakdown.

    Args:
        hypothesis: Candidate translation.
        reference: Reference translation(s).
        max_n: Maximum n-gram order (default 4 for BLEU-4).
        smoothing: Apply smoothing for sentence-level BLEU.

    Returns:
        Dictionary with BLEU score and per-n-gram precisions.
    """
    if isinstance(reference, str):
        reference = [reference]

    hyp_tokens = _tokenize(hypothesis)
    ref_tokens_list = [_tokenize(ref) for ref in reference]

    precisions = []
    for n in range(1, max_n + 1):
        p = _modified_precision(hyp_tokens, ref_tokens_list, n, smoothing)
        precisions.append(p)

    # Brevity penalty
    bp = _brevity_penalty(hyp_tokens, ref_tokens_list)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        log_avg = sum(math.log(p) for p in precisions) / max_n
        bleu = bp * math.exp(log_avg)

    return {
        "bleu": round(bleu, 4),
        "brevity_penalty": round(bp, 4),
        "precisions": [round(p, 4) for p in precisions],
        "hyp_length": len(hyp_tokens),
        "ref_length": _closest_ref_length(hyp_tokens, ref_tokens_list),
    }


def sentence_bleu(
    hypothesis: str,
    reference: str | list[str],
    max_n: int = 4,
) -> float:
    """Compute sentence-level BLEU score.

    Args:
        hypothesis: Candidate translation.
        reference: Reference translation(s).
        max_n: Maximum n-gram order.

    Returns:
        BLEU score as float (0 to 1).
    """
    result = compute_bleu(hypothesis, reference, max_n=max_n, smoothing=True)
    return result["bleu"]


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str | list[str]],
    max_n: int = 4,
) -> float:
    """Compute corpus-level BLEU score.

    Args:
        hypotheses: List of candidate translations.
        references: List of reference translation(s).
        max_n: Maximum n-gram order.

    Returns:
        Corpus BLEU score.
    """
    assert len(hypotheses) == len(references), "Mismatched hypothesis/reference counts"

    total_hyp_length = 0
    total_ref_length = 0
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    for hyp, ref in zip(hypotheses, references):
        if isinstance(ref, str):
            ref = [ref]

        hyp_tokens = _tokenize(hyp)
        ref_tokens_list = [_tokenize(r) for r in ref]

        total_hyp_length += len(hyp_tokens)
        total_ref_length += _closest_ref_length(hyp_tokens, ref_tokens_list)

        for n in range(1, max_n + 1):
            hyp_ngrams = _get_ngrams(hyp_tokens, n)
            max_ref_ngrams = _max_ref_ngrams(ref_tokens_list, n)

            for ngram, count in hyp_ngrams.items():
                clipped_counts[n - 1] += min(count, max_ref_ngrams.get(ngram, 0))
                total_counts[n - 1] += count

    # Compute precisions
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    # Brevity penalty
    if total_hyp_length == 0:
        return 0.0

    bp = 1.0
    if total_hyp_length < total_ref_length:
        bp = math.exp(1 - total_ref_length / total_hyp_length)

    # BLEU score
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    return round(bp * math.exp(log_avg), 4)


def _modified_precision(
    hypothesis: list[str],
    references: list[list[str]],
    n: int,
    smoothing: bool = False,
) -> float:
    """Compute modified n-gram precision."""
    hyp_ngrams = _get_ngrams(hypothesis, n)
    max_ref_ngrams = _max_ref_ngrams(references, n)

    clipped = 0
    total = 0

    for ngram, count in hyp_ngrams.items():
        clipped += min(count, max_ref_ngrams.get(ngram, 0))
        total += count

    if total == 0:
        return 0.0

    if smoothing and clipped == 0:
        # Add-one smoothing for sentence-level BLEU
        return 1.0 / (total + 1)

    return clipped / total


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Extract n-grams from token list."""
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def _max_ref_ngrams(references: list[list[str]], n: int) -> Counter:
    """Get maximum n-gram counts across all references."""
    max_counts: Counter = Counter()
    for ref in references:
        ref_ngrams = _get_ngrams(ref, n)
        for ngram, count in ref_ngrams.items():
            max_counts[ngram] = max(max_counts[ngram], count)
    return max_counts


def _brevity_penalty(
    hypothesis: list[str],
    references: list[list[str]],
) -> float:
    """Compute brevity penalty."""
    hyp_len = len(hypothesis)
    ref_len = _closest_ref_length(hypothesis, references)

    if hyp_len == 0:
        return 0.0
    if hyp_len >= ref_len:
        return 1.0
    return math.exp(1 - ref_len / hyp_len)


def _closest_ref_length(
    hypothesis: list[str],
    references: list[list[str]],
) -> int:
    """Find the reference length closest to hypothesis length."""
    hyp_len = len(hypothesis)
    ref_lengths = [len(ref) for ref in references]
    return min(ref_lengths, key=lambda r: (abs(r - hyp_len), r))


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization with lowercasing."""
    return text.lower().split()
