"""Tests for dkmem.memory.gate (Task 19: deterministic DK-Mem distinction gating).

Run: python -m unittest discover -s tests
"""

import unittest
from pathlib import Path

from dkmem.memory.extract import derive_lang_profile
from dkmem.memory.gate import (
    COMPATIBILITY,
    DEFAULT_DISCRIMINATIVE_FEATURES,
    build_merge_event,
    compatible,
    gate,
)
from dkmem.memory.schema import DECISIONS, Extraction, MergeEvent, read_jsonl

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def real_extraction(pair_id, side, **distinction):
    """A minimal, schema-valid Extraction carrying the given distinction dict."""
    surface = "utterance"
    return Extraction(
        pair_id=pair_id,
        side=side,
        raw_output="{}",
        gloss="gloss",
        distinction=distinction,
        surface=surface,
        lang_profile=derive_lang_profile("hi", surface),
        backbone="fake/backbone",
        prompt_id="mem0_extraction_v1",
        seed=0,
    )


class TestCompatibleThreeCases(unittest.TestCase):
    """The three cases named in the task, using its own kinship example."""

    def test_compatible_when_values_agree(self):
        self.assertEqual(compatible({"kinship": "chachi"}, {"kinship": "chachi"}), "compatible")

    def test_compatible_when_neither_side_marks_the_feature(self):
        self.assertEqual(compatible({}, {}), "compatible")
        self.assertEqual(compatible({"name_variant": "Priya-latin"},
                                    {"name_variant": "Priya-devanagari"}), "compatible")

    def test_incompatible_bua_vs_mausi(self):
        self.assertEqual(compatible({"kinship": "bua"}, {"kinship": "mausi"}), "incompatible")

    def test_underdetermined_one_marked_one_not(self):
        self.assertEqual(compatible({"kinship": "chachi"}, {}), "underdetermined")
        self.assertEqual(compatible({}, {"kinship": "chachi"}), "underdetermined")

    def test_return_value_is_one_of_the_declared_constant(self):
        for a, b in (({"kinship": "bua"}, {"kinship": "bua"}),
                    ({"kinship": "bua"}, {"kinship": "mausi"}),
                    ({"kinship": "bua"}, {})):
            self.assertIn(compatible(a, b), COMPATIBILITY)


class TestCompatibleGeneral(unittest.TestCase):
    def test_non_discriminative_feature_disagreeing_is_still_compatible(self):
        # name_variant is not in DEFAULT_DISCRIMINATIVE_FEATURES.
        self.assertEqual(
            compatible({"name_variant": "Priya-latin"}, {"name_variant": "Priya-devanagari"}),
            "compatible",
        )

    def test_custom_discriminative_features_override_default(self):
        # With an empty override, nothing gates: even a real conflict passes.
        self.assertEqual(
            compatible({"kinship": "bua"}, {"kinship": "mausi"}, discriminative_features=set()),
            "compatible",
        )
        # Promoting name_variant to discriminative makes its conflict count.
        self.assertEqual(
            compatible({"name_variant": "Priya-latin"}, {"name_variant": "Priya-devanagari"},
                      discriminative_features={"name_variant"}),
            "incompatible",
        )

    def test_incompatible_outranks_underdetermined_across_features(self):
        # kinship: incompatible; register: underdetermined. Incompatible wins.
        a = {"kinship": "bua", "register": "tu"}
        b = {"kinship": "mausi"}
        self.assertEqual(compatible(a, b), "incompatible")

    def test_default_discriminative_features_match_agreed_set(self):
        self.assertEqual(
            DEFAULT_DISCRIMINATIVE_FEATURES,
            {"kinship", "register", "classifier", "evidentiality", "politeness", "temporal_deixis"},
        )


class TestGateDecision(unittest.TestCase):
    def test_compatible_merges_when_sim_meets_tau(self):
        compatibility, decision, reason = gate({"kinship": "chachi"}, {"kinship": "chachi"},
                                               sim=0.9, tau=0.85)
        self.assertEqual((compatibility, decision), ("compatible", "merge"))
        self.assertTrue(reason.startswith("compatible:"))

    def test_compatible_keeps_both_when_sim_below_tau(self):
        compatibility, decision, _ = gate({"kinship": "chachi"}, {"kinship": "chachi"},
                                          sim=0.5, tau=0.85)
        self.assertEqual((compatibility, decision), ("compatible", "keep_both"))

    def test_compatible_sim_equal_tau_merges(self):
        _, decision, _ = gate({}, {}, sim=0.85, tau=0.85)
        self.assertEqual(decision, "merge")

    def test_incompatible_keeps_both_regardless_of_sim(self):
        for sim in (0.99, 0.0):
            with self.subTest(sim=sim):
                compatibility, decision, reason = gate(
                    {"kinship": "bua"}, {"kinship": "mausi"}, sim=sim, tau=0.85
                )
                self.assertEqual((compatibility, decision), ("incompatible", "keep_both"))
                self.assertEqual(reason, "incompatible kinship values: bua vs mausi")

    def test_underdetermined_links_regardless_of_sim(self):
        for sim in (0.99, 0.0):
            with self.subTest(sim=sim):
                compatibility, decision, reason = gate(
                    {"kinship": "chachi"}, {}, sim=sim, tau=0.85
                )
                self.assertEqual((compatibility, decision), ("underdetermined", "link_unresolved"))
                self.assertEqual(reason, "kinship marked (chachi) on one side, unmarked on the other")

    def test_decision_never_supersede(self):
        cases = [
            ({"kinship": "chachi"}, {"kinship": "chachi"}, 1.0, 0.85),
            ({"kinship": "chachi"}, {"kinship": "chachi"}, 0.0, 0.85),
            ({"kinship": "bua"}, {"kinship": "mausi"}, 1.0, 0.85),
            ({"kinship": "chachi"}, {}, 1.0, 0.85),
        ]
        for a, b, sim, tau in cases:
            _, decision, _ = gate(a, b, sim, tau)
            self.assertIn(decision, DECISIONS)
            self.assertNotEqual(decision, "supersede")


class TestBuildMergeEvent(unittest.TestCase):
    def test_builds_valid_merge_event(self):
        ev = build_merge_event(
            "pair_1", {"kinship": "bua"}, {"kinship": "mausi"},
            sim=0.9, tau=0.85, policy="dkmem_gate_v1", backbone="fake/backbone",
            seed=0, prompt_id="mem0_extraction_v1",
        )
        self.assertIsInstance(ev, MergeEvent)
        self.assertEqual(ev.decision, "keep_both")
        self.assertEqual(ev.pair_id, "pair_1")
        self.assertEqual(ev.tau, 0.85)
        self.assertEqual(ev.sim, 0.9)

    def test_roundtrips_through_jsonl(self):
        ev = build_merge_event(
            "pair_2", {}, {}, sim=0.9, tau=0.85, policy="dkmem_gate_v1",
            backbone="fake/backbone", seed=0, prompt_id="mem0_extraction_v1",
        )
        self.assertEqual(MergeEvent.from_json(ev.to_json()), ev)

    def test_uses_only_existing_schema_fields(self):
        ev = build_merge_event(
            "pair_3", {"kinship": "chachi"}, {}, sim=0.5, tau=0.85,
            policy="dkmem_gate_v1", backbone="fake/backbone", seed=0,
            prompt_id="mem0_extraction_v1",
        )
        self.assertEqual(
            set(ev.to_dict()),
            {"pair_id", "policy", "tau", "sim", "decision", "reason", "backbone", "seed", "prompt_id"},
        )


class TestOperatesOnExtractionNotGoldLabels(unittest.TestCase):
    """Runtime input must be Extraction.distinction, not ProbeItem gold labels."""

    def test_wired_from_real_extraction_objects(self):
        ext_a = real_extraction("p", "a", kinship="bua")
        ext_b = real_extraction("p", "b", kinship="mausi")
        self.assertIsInstance(ext_a, Extraction)
        compatibility, decision, _ = gate(ext_a.distinction, ext_b.distinction, sim=0.95, tau=0.85)
        self.assertEqual((compatibility, decision), ("incompatible", "keep_both"))

    def test_compatible_accepts_plain_extraction_distinction_type(self):
        # Extraction.distinction is dict[str, str]; compatible() must accept
        # exactly that shape without any adaptation.
        ext = real_extraction("p", "a", kinship="chachi")
        self.assertIsInstance(ext.distinction, dict)
        self.assertEqual(compatible(ext.distinction, ext.distinction), "compatible")


class TestReproducesFixtureConvention(unittest.TestCase):
    """gate() must reproduce, pair for pair, the already-agreed dkmem_v1
    decisions recorded in the synthetic fixtures (tests/fixtures/)."""

    def test_matches_all_ten_fixture_probes(self):
        extractions = {}
        for ext in read_jsonl(FIXTURE_DIR / "extractions.jsonl", Extraction):
            extractions.setdefault(ext.pair_id, {})[ext.side] = ext
        fixture_events = {
            ev.pair_id: ev
            for ev in read_jsonl(FIXTURE_DIR / "merge_events.jsonl", MergeEvent)
            if ev.policy == "dkmem_v1"
        }
        self.assertEqual(len(fixture_events), 10)

        for pair_id, expected in fixture_events.items():
            with self.subTest(pair_id=pair_id):
                a = extractions[pair_id]["a"].distinction
                b = extractions[pair_id]["b"].distinction
                # decision is the actual agreed behavior and must match
                # exactly; reason is free-form documentation (the fixtures
                # themselves use different wording per probe even for the
                # same decision, e.g. "temporal deixis" vs the raw key
                # "temporal_deixis"), so it is not asserted here.
                _compatibility, decision, _reason = gate(a, b, expected.sim, expected.tau)
                self.assertEqual(decision, expected.decision)


if __name__ == "__main__":
    unittest.main()
