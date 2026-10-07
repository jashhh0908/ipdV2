"""Tests for dkmem.memory.similarity (candidate matching + pluggable similarity).

The similarity-to-gate glue lives in dkmem.memory.consolidation and is tested
in tests/test_consolidation.py.

Run: python -m unittest discover -s tests
"""

import ast
import unittest
from pathlib import Path

import dkmem.memory.similarity as similarity_module
from dkmem.memory.extract import derive_lang_profile
from dkmem.memory.schema import Extraction
from dkmem.memory.similarity import (
    DEFAULT_SIMILARITY,
    DIFFLIB_RATIO,
    SimilarityError,
    SimilarityMetric,
    candidate_pairs,
    entry_similarity,
    gloss_similarity,
    normalize_gloss,
)


def make_extraction(pair_id, side, gloss, distinction=None, **overrides):
    surface = "utterance"
    defaults = dict(
        pair_id=pair_id,
        side=side,
        raw_output="{}",
        gloss=gloss,
        distinction=distinction or {},
        surface=surface,
        lang_profile=derive_lang_profile("hi", surface),
        backbone="fake/backbone",
        prompt_id="mem0_extraction_v1",
        seed=0,
    )
    defaults.update(overrides)
    return Extraction(**defaults)


class TestNormalizeGloss(unittest.TestCase):
    def test_casefolds_and_collapses_whitespace(self):
        self.assertEqual(normalize_gloss("  User's  Aunt   lives\tin Mumbai "),
                         "user's aunt lives in mumbai")

    def test_idempotent(self):
        g = normalize_gloss("Some Gloss")
        self.assertEqual(normalize_gloss(g), g)


class TestGlossSimilarity(unittest.TestCase):
    def test_identical_glosses_are_1(self):
        self.assertEqual(gloss_similarity("user's aunt lives in Mumbai",
                                          "user's aunt lives in Mumbai"), 1.0)

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(
            gloss_similarity("user's aunt lives in Mumbai", "USER'S AUNT   LIVES IN MUMBAI"), 1.0
        )

    def test_completely_different_glosses_score_low(self):
        self.assertLess(gloss_similarity("user's aunt lives in Mumbai",
                                         "the weather is nice today"), 0.5)

    def test_similar_but_not_identical_is_between_0_and_1(self):
        sim = gloss_similarity("user's aunt lives in Mumbai", "user's aunt moved to Pune")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_symmetric(self):
        a, b = "user's aunt lives in Mumbai", "user's aunt moved to Pune"
        self.assertEqual(gloss_similarity(a, b), gloss_similarity(b, a))

    def test_deterministic_across_calls(self):
        a, b = "user's aunt lives in Mumbai", "user's aunt moved to Pune"
        self.assertEqual(gloss_similarity(a, b), gloss_similarity(a, b))

    def test_empty_glosses(self):
        self.assertEqual(gloss_similarity("", ""), 1.0)
        self.assertEqual(gloss_similarity("something", ""), 0.0)

    def test_in_unit_range(self):
        for a, b in (("x", "y"), ("", "abc"), ("abc def", "abc def ghi")):
            sim = gloss_similarity(a, b)
            self.assertGreaterEqual(sim, 0.0)
            self.assertLessEqual(sim, 1.0)


class TestSimilarityMetric(unittest.TestCase):
    def test_default_is_the_difflib_placeholder(self):
        self.assertIs(DEFAULT_SIMILARITY, DIFFLIB_RATIO)
        self.assertEqual(DIFFLIB_RATIO.name, "difflib_ratio_v1")
        a, b = "user's aunt lives in Mumbai", "user's aunt moved to Pune"
        self.assertEqual(DIFFLIB_RATIO(a, b), gloss_similarity(a, b))

    def test_wraps_any_text_function(self):
        metric = SimilarityMetric("exact_match", lambda a, b: 1.0 if a == b else 0.0)
        self.assertEqual(metric("x", "x"), 1.0)
        self.assertEqual(metric("x", "y"), 0.0)

    def test_integer_scores_become_float(self):
        self.assertIsInstance(SimilarityMetric("const", lambda a, b: 1)("a", "b"), float)

    def test_out_of_range_or_non_finite_scores_are_rejected_not_clipped(self):
        for bad in (-0.01, 1.01, -1.0, float("nan"), float("inf"), "0.5", None, True):
            with self.subTest(bad=bad):
                metric = SimilarityMetric("bad", lambda a, b, bad=bad: bad)
                with self.assertRaises(SimilarityError):
                    metric("a", "b")

    def test_boundaries_allowed(self):
        self.assertEqual(SimilarityMetric("zero", lambda a, b: 0.0)("a", "b"), 0.0)
        self.assertEqual(SimilarityMetric("one", lambda a, b: 1.0)("a", "b"), 1.0)

    def test_name_and_callable_validated(self):
        with self.assertRaises(ValueError):
            SimilarityMetric("", lambda a, b: 1.0)
        with self.assertRaises(ValueError):
            SimilarityMetric("  ", lambda a, b: 1.0)
        with self.assertRaises(TypeError):
            SimilarityMetric("m", "not callable")


class TestEntrySimilarity(unittest.TestCase):
    def setUp(self):
        self.a = make_extraction("p", "a", "user's aunt lives in Mumbai", surface="meri chachi Mumbai mein rehti hai")
        self.b = make_extraction("p", "b", "user's aunt lives in Mumbai", surface="meri mausi Mumbai mein rehti hai")

    def test_defaults_to_gloss_with_the_placeholder_metric(self):
        self.assertEqual(entry_similarity(self.a, self.b), 1.0)
        self.assertEqual(entry_similarity(self.a, self.b), gloss_similarity(self.a.gloss, self.b.gloss))

    def test_surface_field(self):
        sim = entry_similarity(self.a, self.b, text_field="surface")
        self.assertEqual(sim, gloss_similarity(self.a.surface, self.b.surface))
        self.assertLess(sim, 1.0)

    def test_metric_is_swappable_without_touching_callers(self):
        seen = []

        def fake_embedding(x, y):
            seen.append((x, y))
            return 0.25

        metric = SimilarityMetric("fake_embedding_v0", fake_embedding)
        self.assertEqual(entry_similarity(self.a, self.b, metric), 0.25)
        self.assertEqual(seen, [(self.a.gloss, self.b.gloss)])

    def test_unknown_text_field_rejected(self):
        for bad in ("distinction", "raw_output", "Gloss", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                entry_similarity(self.a, self.b, text_field=bad)


class TestSimilarityDoesNotDependOnTheGate(unittest.TestCase):
    def test_similarity_module_imports_nothing_from_gate_or_consolidation(self):
        tree = ast.parse(Path(similarity_module.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        self.assertNotIn("dkmem.memory.gate", imported)
        self.assertNotIn("dkmem.memory.consolidation", imported)

    def test_evaluate_pair_no_longer_lives_here(self):
        self.assertFalse(hasattr(similarity_module, "evaluate_pair"))


class TestCandidatePairs(unittest.TestCase):
    def test_single_entry_each_side_is_one_pair(self):
        a = make_extraction("p1", "a", "g1")
        b = make_extraction("p1", "b", "g2")
        self.assertEqual(candidate_pairs([a], [b]), [(a, b)])

    def test_cross_product_excludes_same_side_pairs(self):
        a1 = make_extraction("p1", "a", "g1")
        a2 = make_extraction("p1", "a", "g2")
        b1 = make_extraction("p1", "b", "g3")
        pairs = candidate_pairs([a1, a2], [b1])
        self.assertEqual(pairs, [(a1, b1), (a2, b1)])
        # No (a1, a2) or (a2, a1) same-side pair appears.
        sides = {(p[0].side, p[1].side) for p in pairs}
        self.assertEqual(sides, {("a", "b")})

    def test_empty_either_side_gives_no_candidates(self):
        b = make_extraction("p1", "b", "g")
        self.assertEqual(candidate_pairs([], [b]), [])
        self.assertEqual(candidate_pairs([], []), [])

    def test_rejects_mismatched_pair_ids(self):
        a = make_extraction("p1", "a", "g1")
        b = make_extraction("p2", "b", "g2")
        with self.assertRaises(SimilarityError):
            candidate_pairs([a], [b])

    def test_rejects_wrong_side_in_either_list(self):
        wrong_a = make_extraction("p1", "b", "g1")  # side b, but passed as entries_a
        b = make_extraction("p1", "b", "g2")
        with self.assertRaises(SimilarityError):
            candidate_pairs([wrong_a], [b])

        a = make_extraction("p1", "a", "g1")
        wrong_b = make_extraction("p1", "a", "g2")  # side a, but passed as entries_b
        with self.assertRaises(SimilarityError):
            candidate_pairs([a], [wrong_b])


if __name__ == "__main__":
    unittest.main()
