"""Tests for dkmem.memory.consolidation (similarity metric + gate -> MergeEvent).

Run: python -m unittest discover -s tests
"""

import unittest
from pathlib import Path

from dkmem.memory.consolidation import evaluate_pair
from dkmem.memory.extract import derive_lang_profile
from dkmem.memory.schema import DECISIONS, Extraction, MergeEvent, read_jsonl
from dkmem.memory.scope import LEGACY_DISCRIMINATIVE_CLASSES
from dkmem.memory.similarity import (
    SimilarityError,
    SimilarityMetric,
    candidate_pairs,
    gloss_similarity,
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


class TestInjectedSimilarity(unittest.TestCase):
    """The metric is a parameter: swapping it changes ``sim`` and nothing else."""

    def setUp(self):
        self.a = make_extraction("p1", "a", "user's aunt lives in Mumbai", {"kinship": "chachi"},
                                 surface="meri chachi Mumbai mein rehti hai")
        self.b = make_extraction("p1", "b", "user's aunt lives in Mumbai", {"kinship": "chachi"},
                                 surface="meri chachi Mumbai mein rehti hai")

    def test_custom_metric_supplies_sim(self):
        low = SimilarityMetric("always_low", lambda x, y: 0.1)
        ev = evaluate_pair(self.a, self.b, tau=0.5, policy="p", similarity=low)
        self.assertEqual(ev.sim, 0.1)
        self.assertEqual(ev.decision, "keep_both")  # compatible, but below tau

        high = SimilarityMetric("always_high", lambda x, y: 0.9)
        ev = evaluate_pair(self.a, self.b, tau=0.5, policy="p", similarity=high)
        self.assertEqual(ev.sim, 0.9)
        self.assertEqual(ev.decision, "merge")

    def test_gate_outcome_is_independent_of_the_metric(self):
        incompatible_b = make_extraction("p1", "b", "user's aunt lives in Mumbai", {"kinship": "mausi"})
        for score in (0.0, 0.5, 1.0):
            with self.subTest(score=score):
                metric = SimilarityMetric("const", lambda x, y, s=score: s)
                ev = evaluate_pair(self.a, incompatible_b, tau=0.5, policy="p", similarity=metric)
                self.assertEqual(ev.sim, score)
                self.assertEqual(ev.decision, "keep_both")

    def test_text_field_selects_what_is_compared(self):
        seen = []

        def spy(x, y):
            seen.append((x, y))
            return 1.0

        metric = SimilarityMetric("spy", spy)
        evaluate_pair(self.a, self.b, tau=0.5, policy="p", similarity=metric)
        evaluate_pair(self.a, self.b, tau=0.5, policy="p", similarity=metric, text_field="surface")
        self.assertEqual(seen, [(self.a.gloss, self.b.gloss), (self.a.surface, self.b.surface)])

    def test_out_of_range_metric_is_rejected_not_clipped(self):
        cosine_like = SimilarityMetric("raw_cosine", lambda x, y: -0.3)
        with self.assertRaises(SimilarityError):
            evaluate_pair(self.a, self.b, tau=0.5, policy="p", similarity=cosine_like)

    def test_discriminative_features_can_be_overridden(self):
        a = make_extraction("p1", "a", "g", {"evidentiality": "reported/hearsay"})
        b = make_extraction("p1", "b", "g", {"evidentiality": "direct/confirmed"})
        self.assertEqual(evaluate_pair(a, b, tau=0.5, policy="p").decision, "merge")  # out of scope
        legacy = evaluate_pair(a, b, tau=0.5, policy="p",
                               discriminative_features=LEGACY_DISCRIMINATIVE_CLASSES)
        self.assertEqual(legacy.decision, "keep_both")


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
