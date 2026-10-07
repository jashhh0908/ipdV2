"""Tests for dkmem.memory.gate (Task 19: deterministic DK-Mem distinction gating).

Run: python -m unittest discover -s tests
"""

import unittest
from pathlib import Path

from dkmem.memory.extract import derive_lang_profile
from dkmem.memory.gate import (
    COMPATIBILITY,
    DEFAULT_DISCRIMINATIVE_FEATURES,
    MERGING_DECISIONS,
    GateResult,
    apply_gate,
    build_merge_event,
    compatible,
    gate,
)
from dkmem.memory.schema import DECISIONS, Extraction, MergeEvent, read_jsonl
from dkmem.memory.scope import LEGACY_DISCRIMINATIVE_CLASSES

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

    def test_default_discriminative_features_are_the_in_scope_cannot_link_set(self):
        self.assertEqual(DEFAULT_DISCRIMINATIVE_FEATURES, {"kinship", "register"})

    def test_out_of_scope_features_no_longer_gate_by_default(self):
        for feature, a, b in (("evidentiality", "reported/hearsay", "direct/confirmed"),
                              ("politeness", "formal", "informal"),
                              ("classifier", "cup", "long-object"),
                              ("temporal_deixis", "yesterday", "tomorrow")):
            with self.subTest(feature=feature):
                self.assertEqual(compatible({feature: a}, {feature: b}), "compatible")
                self.assertEqual(compatible({feature: a}, {}), "compatible")
                # ...but the legacy set still reproduces the old behaviour.
                self.assertEqual(
                    compatible({feature: a}, {feature: b},
                               discriminative_features=LEGACY_DISCRIMINATIVE_CLASSES),
                    "incompatible",
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
                # The fixtures predate the scope change and cover all seven
                # classes, so they are gated with the legacy feature set.
                _compatibility, decision, _reason = gate(
                    a, b, expected.sim, expected.tau,
                    discriminative_features=LEGACY_DISCRIMINATIVE_CLASSES,
                )
                self.assertEqual(decision, expected.decision)


class TestApplyGate(unittest.TestCase):
    """apply_gate vetoes a *host* mechanism's proposal; it never decides on its own."""

    CHACHI, MAUSI = {"kinship": "chachi"}, {"kinship": "mausi"}

    def test_incompatible_merge_is_vetoed_to_keep_both(self):
        r = apply_gate("merge", self.CHACHI, self.MAUSI)
        self.assertIsInstance(r, GateResult)
        self.assertEqual((r.compatibility, r.proposed, r.decision), ("incompatible", "merge", "keep_both"))
        self.assertTrue(r.vetoed)
        self.assertEqual(r.reason, "vetoed merge: incompatible kinship values: chachi vs mausi")

    def test_incompatible_supersede_is_vetoed_to_keep_both(self):
        # The motivating failure: an UPDATE of "aunt in Pune" by "aunt in Delhi"
        # that would delete the first fact.
        r = apply_gate("supersede", self.CHACHI, self.MAUSI)
        self.assertEqual(r.decision, "keep_both")
        self.assertTrue(r.vetoed)

    def test_underdetermined_merge_becomes_link_unresolved(self):
        for proposed in MERGING_DECISIONS:
            with self.subTest(proposed=proposed):
                r = apply_gate(proposed, self.CHACHI, {})
                self.assertEqual((r.compatibility, r.decision), ("underdetermined", "link_unresolved"))
                self.assertTrue(r.vetoed)
                self.assertEqual(
                    r.reason,
                    f"vetoed {proposed}: kinship marked (chachi) on one side, unmarked on the other",
                )

    def test_compatible_proposal_passes_through_unchanged(self):
        for proposed in MERGING_DECISIONS:
            with self.subTest(proposed=proposed):
                r = apply_gate(proposed, self.CHACHI, self.CHACHI)
                self.assertEqual((r.compatibility, r.decision), ("compatible", proposed))
                self.assertFalse(r.vetoed)
        self.assertEqual(apply_gate("merge", {}, {}).decision, "merge")

    def test_non_merging_host_decisions_are_never_changed(self):
        # No merge proposed, so there is nothing to veto -- and no link is made
        # for a pair the host would not have merged.
        for proposed in ("keep_both", "link_unresolved"):
            for a, b in ((self.CHACHI, self.MAUSI), (self.CHACHI, {}), (self.CHACHI, self.CHACHI)):
                with self.subTest(proposed=proposed, a=a, b=b):
                    r = apply_gate(proposed, a, b)
                    self.assertEqual(r.decision, proposed)
                    self.assertFalse(r.vetoed)

    def test_compatibility_is_reported_even_when_nothing_is_vetoed(self):
        self.assertEqual(apply_gate("keep_both", self.CHACHI, self.MAUSI).compatibility, "incompatible")
        self.assertEqual(apply_gate("keep_both", self.CHACHI, {}).compatibility, "underdetermined")
        self.assertEqual(apply_gate("keep_both", {}, {}).compatibility, "compatible")

    def test_unknown_host_decision_rejected(self):
        for bad in ("no_merge", "merged", "", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                apply_gate(bad, {}, {})

    def test_vetoes_only_on_discriminative_features(self):
        a, b = {"name_variant": "Priya-latin"}, {"name_variant": "Priya-devanagari"}
        self.assertFalse(apply_gate("merge", a, b).vetoed)  # same entity: must still merge
        self.assertTrue(apply_gate("merge", a, b, discriminative_features={"name_variant"}).vetoed)

    def test_every_decision_is_in_the_schema_vocabulary(self):
        for proposed in DECISIONS:
            for a, b in ((self.CHACHI, self.MAUSI), (self.CHACHI, {}), ({}, {})):
                self.assertIn(apply_gate(proposed, a, b).decision, DECISIONS)

    def test_agrees_with_standalone_gate_whenever_the_host_would_merge(self):
        # gate() proposes merge iff sim >= tau; when it does, apply_gate must give
        # the same compatibility and decision.
        cases = [(self.CHACHI, self.MAUSI), (self.CHACHI, {}), ({}, self.MAUSI),
                 (self.CHACHI, self.CHACHI), ({}, {}),
                 ({"kinship": "bua", "register": "tu"}, {"kinship": "mausi"})]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                compat, decision, _ = gate(a, b, sim=0.95, tau=0.85)
                r = apply_gate("merge", a, b)
                self.assertEqual((r.compatibility, r.decision), (compat, decision))

    def test_differs_from_standalone_gate_only_below_tau_for_underdetermined(self):
        # gate() links an underdetermined pair even far below tau; the veto layer
        # does not, because the host would not have merged it.
        _, standalone, _ = gate(self.CHACHI, {}, sim=0.1, tau=0.85)
        self.assertEqual(standalone, "link_unresolved")
        self.assertEqual(apply_gate("keep_both", self.CHACHI, {}).decision, "keep_both")
        # Incompatible pairs agree at any similarity.
        _, standalone, _ = gate(self.CHACHI, self.MAUSI, sim=0.1, tau=0.85)
        self.assertEqual(standalone, apply_gate("keep_both", self.CHACHI, self.MAUSI).decision)

    def test_wired_from_real_extraction_objects(self):
        ext_a = real_extraction("p", "a", kinship="chachi")
        ext_b = real_extraction("p", "b", kinship="mausi")
        r = apply_gate("merge", ext_a.distinction, ext_b.distinction)
        self.assertEqual(r.decision, "keep_both")


if __name__ == "__main__":
    unittest.main()
