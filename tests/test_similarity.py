"""Tests for dkmem.memory.similarity (candidate matching + gloss similarity).

Run: python -m unittest discover -s tests
"""

import unittest
from pathlib import Path

from dkmem.memory.extract import derive_lang_profile
from dkmem.memory.schema import DECISIONS, Extraction, MergeEvent, read_jsonl
from dkmem.memory.similarity import (
    SimilarityError,
    candidate_pairs,
    evaluate_pair,
    gloss_similarity,
    normalize_gloss,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


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


class TestEvaluatePair(unittest.TestCase):
    def test_returns_merge_event_with_computed_sim(self):
        a = make_extraction("p1", "a", "user's aunt lives in Mumbai", {"kinship": "chachi"})
        b = make_extraction("p1", "b", "user's aunt lives in Mumbai", {"kinship": "chachi"})
        ev = evaluate_pair(a, b, tau=0.5, policy="dkmem_v1")
        self.assertIsInstance(ev, MergeEvent)
        self.assertEqual(ev.sim, gloss_similarity(a.gloss, b.gloss))
        self.assertEqual(ev.sim, 1.0)
        self.assertEqual(ev.decision, "merge")
        self.assertEqual(ev.pair_id, "p1")
        self.assertEqual(ev.tau, 0.5)
        self.assertEqual(ev.policy, "dkmem_v1")
        self.assertEqual(ev.backbone, "fake/backbone")
        self.assertEqual(ev.seed, 0)
        self.assertEqual(ev.prompt_id, "mem0_extraction_v1")

    def test_incompatible_overrides_high_similarity(self):
        a = make_extraction("p1", "a", "user's aunt lives in Mumbai", {"kinship": "bua"})
        b = make_extraction("p1", "b", "user's aunt lives in Mumbai", {"kinship": "mausi"})
        ev = evaluate_pair(a, b, tau=0.5, policy="dkmem_v1")
        self.assertEqual(ev.sim, 1.0)  # similarity is raw/pre-gating: not clipped
        self.assertEqual(ev.decision, "keep_both")

    def test_underdetermined_overrides_high_similarity(self):
        a = make_extraction("p1", "a", "user's aunt lives in Mumbai", {"kinship": "chachi"})
        b = make_extraction("p1", "b", "user's aunt lives in Mumbai", {})
        ev = evaluate_pair(a, b, tau=0.5, policy="dkmem_v1")
        self.assertEqual(ev.sim, 1.0)
        self.assertEqual(ev.decision, "link_unresolved")

    def test_low_similarity_below_tau_keeps_both(self):
        a = make_extraction("p1", "a", "user's aunt lives in Mumbai", {})
        b = make_extraction("p1", "b", "completely unrelated fact", {})
        ev = evaluate_pair(a, b, tau=0.9, policy="dkmem_v1")
        self.assertLess(ev.sim, 0.9)
        self.assertEqual(ev.decision, "keep_both")

    def test_decision_never_supersede(self):
        a = make_extraction("p1", "a", "g")
        b = make_extraction("p1", "b", "g")
        ev = evaluate_pair(a, b, tau=0.0, policy="dkmem_v1")
        self.assertIn(ev.decision, DECISIONS)
        self.assertNotEqual(ev.decision, "supersede")

    def test_rejects_mismatched_pair_id(self):
        a = make_extraction("p1", "a", "g")
        b = make_extraction("p2", "b", "g")
        with self.assertRaises(SimilarityError):
            evaluate_pair(a, b, tau=0.5, policy="dkmem_v1")

    def test_rejects_mismatched_backbone(self):
        a = make_extraction("p1", "a", "g", backbone="model-1")
        b = make_extraction("p1", "b", "g", backbone="model-2")
        with self.assertRaises(SimilarityError):
            evaluate_pair(a, b, tau=0.5, policy="dkmem_v1")

    def test_rejects_mismatched_seed(self):
        a = make_extraction("p1", "a", "g", seed=0)
        b = make_extraction("p1", "b", "g", seed=1)
        with self.assertRaises(SimilarityError):
            evaluate_pair(a, b, tau=0.5, policy="dkmem_v1")

    def test_rejects_mismatched_prompt_id(self):
        a = make_extraction("p1", "a", "g", prompt_id="mem0_extraction_v1")
        b = make_extraction("p1", "b", "g", prompt_id="mem0_extraction_v1+dkmem_lexicon")
        with self.assertRaises(SimilarityError):
            evaluate_pair(a, b, tau=0.5, policy="dkmem_v1")

    def test_roundtrips_through_jsonl(self):
        a = make_extraction("p1", "a", "g")
        b = make_extraction("p1", "b", "g")
        ev = evaluate_pair(a, b, tau=0.5, policy="dkmem_v1")
        self.assertEqual(MergeEvent.from_json(ev.to_json()), ev)


class TestEndToEndOnRealFixtures(unittest.TestCase):
    """ProbeItem -> Extraction (fixture) -> similarity -> gate -> MergeEvent,
    exercised on the real fixture Extractions (not the withheld Tier 1 file)."""

    def test_full_flow_on_fixture_extractions(self):
        by_pair = {}
        for ext in read_jsonl(FIXTURE_DIR / "extractions.jsonl", Extraction):
            by_pair.setdefault(ext.pair_id, {})[ext.side] = ext

        for pair_id, sides in by_pair.items():
            with self.subTest(pair_id=pair_id):
                a, b = sides["a"], sides["b"]
                pairs = candidate_pairs([a], [b])
                self.assertEqual(len(pairs), 1)
                ev = evaluate_pair(a, b, tau=0.85, policy="dkmem_v1")
                self.assertIsInstance(ev, MergeEvent)
                self.assertIn(ev.decision, DECISIONS)
                self.assertNotEqual(ev.decision, "supersede")
                # sim is raw/pre-gating: recomputable independently of the decision.
                self.assertEqual(ev.sim, gloss_similarity(a.gloss, b.gloss))


if __name__ == "__main__":
    unittest.main()
