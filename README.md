# MT Eval Toolkit

A machine translation evaluation toolkit implementing standard automatic metrics. Compute BLEU, chrF, TER, and BERTScore for translation quality assessment.

## Features

- **BLEU** -- bilingual evaluation understudy (sentence and corpus level)
- **chrF** -- character n-gram F-score
- **TER** -- translation edit rate
- **BERTScore** -- contextual embedding similarity
- **Multi-reference** -- evaluate against multiple reference translations
- **Batch evaluation** -- score entire test sets efficiently
- **Report generation** -- formatted comparison reports

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from mt_eval import MTEvaluator

evaluator = MTEvaluator()

hypothesis = "The cat sat on the mat."
reference = "The cat is sitting on the mat."

scores = evaluator.evaluate(hypothesis, reference)
print(scores)
# {'bleu': 0.4647, 'chrf': 0.7823, 'ter': 0.2857}
```

## CLI Usage

```bash
# Evaluate a hypothesis file against a reference
python -m mt_eval evaluate --hyp translations.txt --ref reference.txt

# Compare two systems
python -m mt_eval compare --sys1 system_a.txt --sys2 system_b.txt --ref reference.txt

# Detailed per-sentence scores
python -m mt_eval evaluate --hyp translations.txt --ref reference.txt --detailed
```

## Supported Metrics

| Metric | Level | Range | Higher = Better |
|--------|-------|-------|----------------|
| BLEU | Corpus/Sentence | 0-1 | Yes |
| chrF | Corpus/Sentence | 0-1 | Yes |
| TER | Corpus/Sentence | 0-inf | No |
| BERTScore | Sentence | 0-1 | Yes |

## License

MIT
