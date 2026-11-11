"""Unified MT evaluation interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mt_eval.bleu import compute_bleu, corpus_bleu
from mt_eval.chrf import compute_chrf, corpus_chrf
from mt_eval.ter import compute_ter, corpus_ter


@dataclass
class EvalResult:
    """Evaluation results for a translation pair."""

    hypothesis: str
    reference: str
    bleu: float
    chrf: float
    ter: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bleu": self.bleu,
            "chrf": self.chrf,
            "ter": self.ter,
        }

    def summary(self) -> str:
        return (
            f"BLEU: {self.bleu:.4f} | "
            f"chrF: {self.chrf:.4f} | "
            f"TER: {self.ter:.4f}"
        )


class MTEvaluator:
    """Unified machine translation evaluator.

    Computes multiple metrics in one call for convenience.

    Example:
        >>> evaluator = MTEvaluator()
        >>> result = evaluator.evaluate("The cat sat.", "The cat is sitting.")
        >>> print(result.summary())
    """

    def __init__(self, metrics: list[str] | None = None):
        """Initialize the evaluator.

        Args:
            metrics: List of metrics to compute.
                     Options: 'bleu', 'chrf', 'ter'.
                     Defaults to all.
        """
        self.metrics = metrics or ["bleu", "chrf", "ter"]

    def evaluate(self, hypothesis: str, reference: str) -> EvalResult:
        """Evaluate a single translation pair.

        Args:
            hypothesis: Candidate translation.
            reference: Reference translation.

        Returns:
            EvalResult with all computed metrics.
        """
        bleu_score = 0.0
        chrf_score = 0.0
        ter_score = 0.0

        if "bleu" in self.metrics:
            bleu_result = compute_bleu(hypothesis, reference)
            bleu_score = bleu_result["bleu"]

        if "chrf" in self.metrics:
            chrf_score = compute_chrf(hypothesis, reference)

        if "ter" in self.metrics:
            ter_score = compute_ter(hypothesis, reference)

        return EvalResult(
            hypothesis=hypothesis,
            reference=reference,
            bleu=bleu_score,
            chrf=chrf_score,
            ter=ter_score,
        )

    def evaluate_corpus(
        self,
        hypotheses: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Evaluate a corpus of translations.

        Args:
            hypotheses: List of candidate translations.
            references: List of reference translations.

        Returns:
            Dictionary of corpus-level scores.
        """
        scores: dict[str, float] = {}

        if "bleu" in self.metrics:
            scores["bleu"] = corpus_bleu(hypotheses, references)

        if "chrf" in self.metrics:
            scores["chrf"] = corpus_chrf(hypotheses, references)

        if "ter" in self.metrics:
            scores["ter"] = corpus_ter(hypotheses, references)

        return scores

    def evaluate_file(
        self,
        hyp_path: str,
        ref_path: str,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """Evaluate translation files line by line.

        Args:
            hyp_path: Path to hypothesis file (one sentence per line).
            ref_path: Path to reference file (one sentence per line).
            encoding: File encoding.

        Returns:
            Dictionary with corpus scores and per-sentence details.
        """
        with open(hyp_path, "r", encoding=encoding) as f:
            hypotheses = [line.strip() for line in f if line.strip()]

        with open(ref_path, "r", encoding=encoding) as f:
            references = [line.strip() for line in f if line.strip()]

        assert len(hypotheses) == len(references), (
            f"Line count mismatch: {len(hypotheses)} hyp vs {len(references)} ref"
        )

        corpus_scores = self.evaluate_corpus(hypotheses, references)

        per_sentence = [
            self.evaluate(hyp, ref).to_dict()
            for hyp, ref in zip(hypotheses, references)
        ]

        return {
            "corpus": corpus_scores,
            "num_sentences": len(hypotheses),
            "per_sentence": per_sentence,
        }

    def compare_systems(
        self,
        systems: dict[str, list[str]],
        references: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compare multiple MT systems.

        Args:
            systems: Dictionary mapping system names to their outputs.
            references: Reference translations.

        Returns:
            Dictionary mapping system names to their corpus scores.
        """
        results = {}
        for name, hypotheses in systems.items():
            results[name] = self.evaluate_corpus(hypotheses, references)
        return results
