"""Tests for word alignment module."""
import pytest
from mt_eval.alignment import WordAligner, CognateMatcher, tokenize


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_with_punctuation(self):
        tokens = tokenize("Hello, World!")
        assert "hello" in tokens
        assert "world" in tokens


class TestWordAligner:
    @pytest.fixture
    def trained_aligner(self):
        aligner = WordAligner()
        aligner.train(
            ["the cat", "the dog", "a cat"],
            ["die Katze", "der Hund", "eine Katze"],
            iterations=10,
        )
        return aligner

    def test_train_sets_flag(self, trained_aligner):
        assert trained_aligner.trained is True

    def test_translation_table(self, trained_aligner):
        table = trained_aligner.get_translation_table(top_k=3)
        assert "cat" in table
        # "katze" should be among top translations
        translations = [t[0] for t in table["cat"]]
        assert "katze" in translations

    def test_align_returns_pairs(self, trained_aligner):
        alignment = trained_aligner.align("the cat", "die Katze")
        assert isinstance(alignment, list)
        assert all(isinstance(a, tuple) and len(a) == 2 for a in alignment)


class TestCognateMatcher:
    def test_identical_words(self):
        sim = CognateMatcher.similarity("computer", "computer")
        assert sim == 1.0

    def test_similar_cognates(self):
        sim = CognateMatcher.similarity("information", "Information")
        assert sim > 0.9

    def test_dissimilar_words(self):
        sim = CognateMatcher.similarity("cat", "Hund")
        assert sim < 0.4

    def test_empty_string(self):
        assert CognateMatcher.similarity("", "hello") == 0.0

    def test_find_cognates(self):
        cognates = CognateMatcher.find_cognates(
            ["computer", "telephone"],
            ["Computer", "Telefon"],
            threshold=0.5,
        )
        assert len(cognates) >= 1
        src_words = [c[0] for c in cognates]
        assert "computer" in src_words
