"""MT Eval Toolkit -- Machine Translation Evaluation Metrics."""

from mt_eval.bleu import compute_bleu, sentence_bleu, corpus_bleu
from mt_eval.chrf import compute_chrf
from mt_eval.ter import compute_ter
from mt_eval.evaluator import MTEvaluator

__version__ = "0.1.0"
__all__ = [
    "MTEvaluator",
    "compute_bleu",
    "sentence_bleu",
    "corpus_bleu",
    "compute_chrf",
    "compute_ter",
]
