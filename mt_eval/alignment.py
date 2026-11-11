"""
Word Alignment Module for Machine Translation Evaluation.
Implements word alignment algorithms used in MT quality estimation
and error analysis. Supports both statistical and heuristic alignment.
"""
import re
from collections import defaultdict
from typing import Optional


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing."""
    return re.findall(r'\w+', text.lower())


class WordAligner:
    """
    Statistical word aligner using IBM Model 1 (simplified).
    Learns word translation probabilities from parallel text.
    """

    def __init__(self):
        self.translation_probs = defaultdict(lambda: defaultdict(float))
        self.trained = False

    def train(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        iterations: int = 10,
    ):
        """Train translation probabilities using EM algorithm (IBM Model 1)."""
        assert len(source_sentences) == len(target_sentences)

        # Tokenize
        src_sents = [tokenize(s) for s in source_sentences]
        tgt_sents = [tokenize(t) for t in target_sentences]

        # Initialize uniform probabilities
        vocab_pairs = set()
        for src, tgt in zip(src_sents, tgt_sents):
            for sw in src:
                for tw in tgt:
                    vocab_pairs.add((sw, tw))

        # Uniform initialization
        for sw, tw in vocab_pairs:
            self.translation_probs[sw][tw] = 1.0

        # EM iterations
        for _ in range(iterations):
            counts = defaultdict(lambda: defaultdict(float))
            total_s = defaultdict(float)

            for src, tgt in zip(src_sents, tgt_sents):
                for tw in tgt:
                    # Compute normalization
                    z = sum(self.translation_probs[sw][tw] for sw in src)
                    z = max(z, 1e-10)

                    for sw in src:
                        c = self.translation_probs[sw][tw] / z
                        counts[sw][tw] += c
                        total_s[sw] += c

            # Normalize
            for sw in counts:
                for tw in counts[sw]:
                    self.translation_probs[sw][tw] = counts[sw][tw] / max(total_s[sw], 1e-10)

        self.trained = True

    def align(self, source: str, target: str) -> list[tuple[int, int]]:
        """Align words between a source and target sentence."""
        src_tokens = tokenize(source)
        tgt_tokens = tokenize(target)
        alignments = []

        for j, tw in enumerate(tgt_tokens):
            best_i = 0
            best_prob = 0.0
            for i, sw in enumerate(src_tokens):
                prob = self.translation_probs[sw].get(tw, 0.0)
                if prob > best_prob:
                    best_prob = prob
                    best_i = i
            if best_prob > 0:
                alignments.append((best_i, j))

        return alignments

    def get_translation_table(self, top_k: int = 5) -> dict[str, list[tuple[str, float]]]:
        """Get the top-k translations for each source word."""
        table = {}
        for sw in self.translation_probs:
            translations = sorted(
                self.translation_probs[sw].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            table[sw] = translations
        return table


class CognateMatcher:
    """
    Match cognates between source and target languages.
    Uses character-level similarity (longest common subsequence ratio).
    """

    @staticmethod
    def lcs_length(s1: str, s2: str) -> int:
        """Compute length of the longest common subsequence."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    @classmethod
    def similarity(cls, word1: str, word2: str) -> float:
        """Compute cognate similarity score between two words."""
        w1, w2 = word1.lower(), word2.lower()
        if not w1 or not w2:
            return 0.0
        lcs = cls.lcs_length(w1, w2)
        return (2 * lcs) / (len(w1) + len(w2))

    @classmethod
    def find_cognates(
        cls,
        source_tokens: list[str],
        target_tokens: list[str],
        threshold: float = 0.6,
    ) -> list[tuple[str, str, float]]:
        """Find cognate pairs between source and target token lists."""
        cognates = []
        for sw in source_tokens:
            best_tw = None
            best_sim = 0.0
            for tw in target_tokens:
                sim = cls.similarity(sw, tw)
                if sim > best_sim:
                    best_sim = sim
                    best_tw = tw
            if best_sim >= threshold and best_tw:
                cognates.append((sw, best_tw, best_sim))
        return cognates


if __name__ == "__main__":
    # Demo: IBM Model 1 alignment
    src_sents = [
        "the cat sits on the mat",
        "the dog runs in the park",
        "the cat runs fast",
        "a dog sits here",
    ]
    tgt_sents = [
        "die Katze sitzt auf der Matte",
        "der Hund laeuft im Park",
        "die Katze laeuft schnell",
        "ein Hund sitzt hier",
    ]

    aligner = WordAligner()
    aligner.train(src_sents, tgt_sents, iterations=15)

    print("=== Translation Table (top 3) ===")
    table = aligner.get_translation_table(top_k=3)
    for word in ["cat", "dog", "sits", "runs"]:
        if word in table:
            print(f"  {word}: {table[word]}")

    print("\n=== Word Alignment ===")
    alignment = aligner.align("the cat sits on the mat", "die Katze sitzt auf der Matte")
    src_tokens = tokenize("the cat sits on the mat")
    tgt_tokens = tokenize("die Katze sitzt auf der Matte")
    for si, ti in alignment:
        print(f"  {src_tokens[si]} -> {tgt_tokens[ti]}")

    print("\n=== Cognate Detection ===")
    cognates = CognateMatcher.find_cognates(
        ["information", "computer", "telephone", "university"],
        ["Information", "Computer", "Telefon", "Universitaet"],
    )
    for src, tgt, sim in cognates:
        print(f"  {src} <-> {tgt} (similarity: {sim:.3f})")
"""
