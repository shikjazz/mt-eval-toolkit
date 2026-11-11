"""Translation Edit Rate (TER) metric.

Implements TER as described in:
Snover et al. (2006) "A Study of Translation Edit Rate with
Targeted Human Annotation"
"""

from __future__ import annotations


def compute_ter(hypothesis: str, reference: str) -> float:
    """Compute Translation Edit Rate.

    TER = (number of edits) / (number of reference words)
    Edits: insertions, deletions, substitutions, and shifts.

    Args:
        hypothesis: Candidate translation.
        reference: Reference translation.

    Returns:
        TER score (0 = perfect, higher = worse).
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    # Compute minimum edit distance
    edit_distance = _edit_distance(hyp_tokens, ref_tokens)

    # Compute shift distance (number of block moves needed)
    shift_cost = _count_shifts(hyp_tokens, ref_tokens)

    total_edits = edit_distance + shift_cost
    ter = total_edits / len(ref_tokens)

    return round(ter, 4)


def corpus_ter(
    hypotheses: list[str],
    references: list[str],
) -> float:
    """Compute corpus-level TER.

    Args:
        hypotheses: List of candidate translations.
        references: List of reference translations.

    Returns:
        Corpus-level TER score.
    """
    assert len(hypotheses) == len(references)

    total_edits = 0
    total_ref_length = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.lower().split()
        ref_tokens = ref.lower().split()

        if ref_tokens:
            edits = _edit_distance(hyp_tokens, ref_tokens)
            shifts = _count_shifts(hyp_tokens, ref_tokens)
            total_edits += edits + shifts
            total_ref_length += len(ref_tokens)

    if total_ref_length == 0:
        return 0.0

    return round(total_edits / total_ref_length, 4)


def _edit_distance(hyp: list[str], ref: list[str]) -> int:
    """Compute minimum edit distance (Levenshtein)."""
    m, n = len(hyp), len(ref)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if hyp[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def _count_shifts(hyp: list[str], ref: list[str]) -> int:
    """Estimate the number of block shifts needed.

    A simplified version that counts matching n-grams
    that appear in different positions.
    """
    shifts = 0

    # Find matching bigrams in different positions
    hyp_bigrams = {}
    for i in range(len(hyp) - 1):
        bigram = (hyp[i], hyp[i + 1])
        if bigram not in hyp_bigrams:
            hyp_bigrams[bigram] = []
        hyp_bigrams[bigram].append(i)

    ref_bigrams = {}
    for i in range(len(ref) - 1):
        bigram = (ref[i], ref[i + 1])
        if bigram not in ref_bigrams:
            ref_bigrams[bigram] = []
        ref_bigrams[bigram].append(i)

    # Count bigrams that match but are in different relative positions
    for bigram in hyp_bigrams:
        if bigram in ref_bigrams:
            hyp_positions = hyp_bigrams[bigram]
            ref_positions = ref_bigrams[bigram]
            for hp in hyp_positions:
                for rp in ref_positions:
                    if abs(hp - rp) > 1:  # Shifted
                        shifts += 1
                        break
                break  # Count each shift once

    return shifts
