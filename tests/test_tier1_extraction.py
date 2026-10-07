"""Tests for dkmem.tier1.extraction (the no-LLM dk-mem-lexicon path).

Run: python -m unittest discover -s tests
"""

import unittest
from pathlib import Path

from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.schema import Extraction
from dkmem.tier1.extraction import (
    LEXICON_ONLY_PROMPT_ID,
    NO_BACKBONE,
    extract_lexicon_only,
)

LEXICON_PATH = Path(__file__).parent.parent / "dkmem" / "memory" / "distinction_features.json"


class TestExtractLexiconOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = load_lexicon(LEXICON_PATH)

    def test_returns_valid_extraction(self):
        ext = extract_lexicon_only("p1", "a", "meri chachi Pune mein rehti hai", "hi", self.lex, seed=0)
        self.assertIsInstance(ext, Extraction)
        self.assertEqual(ext.pair_id, "p1")
        self.assertEqual(ext.side, "a")

    def test_no_llm_sentinels(self):
        ext = extract_lexicon_only("p1", "a", "meri chachi Pune mein rehti hai", "hi", self.lex, seed=0)
        self.assertEqual(ext.backbone, NO_BACKBONE)
        self.assertEqual(ext.prompt_id, LEXICON_ONLY_PROMPT_ID)
        self.assertEqual(ext.raw_output, "")

    def test_gloss_and_surface_are_the_raw_utterance(self):
        utt = "meri chachi Pune mein rehti hai"
        ext = extract_lexicon_only("p1", "a", utt, "hi", self.lex, seed=0)
        self.assertEqual(ext.gloss, utt)
        self.assertEqual(ext.surface, utt)

    def test_kinship_matched_from_real_lexicon(self):
        ext = extract_lexicon_only("p1", "a", "meri chachi Pune mein rehti hai", "hi", self.lex, seed=0)
        self.assertEqual(ext.distinction, {"kinship": "chachi"})

    def test_no_match_gives_empty_distinction(self):
        ext = extract_lexicon_only("p1", "a", "yeh ek ghar hai", "hi", self.lex, seed=0)
        self.assertEqual(ext.distinction, {})

    def test_devanagari_utterance_matched(self):
        ext = extract_lexicon_only("p1", "a", "मेरी चाची मुंबई में रहती है", "hi", self.lex, seed=0)
        self.assertEqual(ext.distinction, {"kinship": "chachi"})

    def test_turkish_evidential_suffix_not_matched_out_of_scope(self):
        ext = extract_lexicon_only("p1", "a", "Ali Ankara'ya gitmiş.", "tr", self.lex, seed=0)
        self.assertEqual(ext.distinction, {})

    def test_code_mixed_lang_checks_both_components(self):
        ext = extract_lexicon_only("p1", "a", "meri chachi call kiya", "hi-en", self.lex, seed=0)
        self.assertEqual(ext.distinction, {"kinship": "chachi"})

    def test_unsupported_lang_raises_value_error(self):
        with self.assertRaises(ValueError):
            extract_lexicon_only("p1", "a", "some text", "not-a-lang-tag!", self.lex, seed=0)

    def test_bad_side_rejected(self):
        with self.assertRaises(ValueError):
            extract_lexicon_only("p1", "c", "text", "hi", self.lex, seed=0)

    def test_seed_is_recorded(self):
        ext = extract_lexicon_only("p1", "a", "text", "hi", self.lex, seed=7)
        self.assertEqual(ext.seed, 7)

    def test_opaque_entity_ids_never_touched(self):
        # extract_lexicon_only's signature has no parameter for gold/entity
        # info at all -- there is nothing to check at runtime beyond the
        # fact that the function cannot accept it.
        import inspect
        params = inspect.signature(extract_lexicon_only).parameters
        self.assertNotIn("opaque_entity_id_a", params)
        self.assertNotIn("gold_entity_id", params)


if __name__ == "__main__":
    unittest.main()
