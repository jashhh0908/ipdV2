"""Validate the synthetic JSONL fixtures in tests/fixtures/.

Run: python -m unittest discover -s tests
"""

import collections
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from dkmem.memory.schema import DECISIONS, Extraction, MergeEvent, ProbeItem, read_jsonl

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "synthetic_fixtures", FIXTURE_DIR / "synthetic_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # read_jsonl re-validates every record against the schema.
        cls.probes = list(read_jsonl(FIXTURE_DIR / "probe_items.jsonl", ProbeItem))
        cls.extractions = list(read_jsonl(FIXTURE_DIR / "extractions.jsonl", Extraction))
        cls.events = list(read_jsonl(FIXTURE_DIR / "merge_events.jsonl", MergeEvent))
        cls.pmap = {p.pair_id: p for p in cls.probes}

    def test_jsonl_is_up_to_date_with_generator(self):
        gen = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            gen.main(Path(tmp))
            for name in ("probe_items", "extractions", "merge_events"):
                with self.subTest(name=name):
                    fresh = (Path(tmp) / f"{name}.jsonl").read_bytes()
                    self.assertEqual(fresh, (FIXTURE_DIR / f"{name}.jsonl").read_bytes())

    def test_counts(self):
        self.assertEqual(len(self.probes), 10)
        self.assertEqual(len(self.extractions), 20)
        self.assertEqual(len(self.events), 40)

    def test_unique_pair_ids(self):
        self.assertEqual(len(self.pmap), len(self.probes))

    def test_each_probe_has_one_extraction_per_side(self):
        sides = collections.defaultdict(list)
        for e in self.extractions:
            sides[e.pair_id].append(e.side)
        self.assertEqual(set(sides), set(self.pmap))
        for pid, s in sides.items():
            self.assertEqual(sorted(s), ["a", "b"], pid)

    def test_extraction_matches_probe(self):
        for e in self.extractions:
            p = self.pmap[e.pair_id]
            utt = p.utt_a if e.side == "a" else p.utt_b
            value = p.distinction_value_a if e.side == "a" else p.distinction_value_b
            with self.subTest(pair=e.pair_id, side=e.side):
                self.assertEqual(e.surface, utt)
                key = p.distinction_class.removeprefix("control_")
                if value is None:
                    self.assertEqual(e.distinction, {})
                else:
                    self.assertEqual(e.distinction, {key: value})

    def test_raw_output_is_json_matching_fields(self):
        for e in self.extractions:
            with self.subTest(pair=e.pair_id, side=e.side):
                parsed = json.loads(e.raw_output)
                self.assertEqual(
                    parsed, {"gloss": e.gloss, "distinction": e.distinction, "surface": e.surface}
                )

    def test_lang_profile_contains_probe_languages(self):
        for e in self.extractions:
            langs = self.pmap[e.pair_id].lang.split("-")
            with self.subTest(pair=e.pair_id, side=e.side):
                for lang in langs:
                    self.assertIn(lang, e.lang_profile)

    def test_merge_events_reference_probes_once_per_policy(self):
        keys = [(m.pair_id, m.policy) for m in self.events]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertLessEqual({m.pair_id for m in self.events}, set(self.pmap))
        policies = {m.policy for m in self.events}
        for pid in self.pmap:
            self.assertEqual({p for q, p in keys if q == pid}, policies, pid)

    def test_coverage(self):
        self.assertEqual({m.decision for m in self.events}, set(DECISIONS))
        gold = {p.gold_same_entity for p in self.probes}
        self.assertEqual(gold, {True, False})
        self.assertTrue(any((p.distinction_value_a is None) != (p.distinction_value_b is None)
                            for p in self.probes), "no underdetermined probe")
        self.assertTrue(any(not p.lang.startswith("hi") for p in self.probes), "no non-Indic probe")

    def test_baseline_policy_rules(self):
        for m in self.events:
            consolidated = m.decision in ("merge", "supersede")
            with self.subTest(pair=m.pair_id, policy=m.policy):
                if m.policy in ("mem0_style", "surface_only"):
                    self.assertEqual(consolidated, m.sim >= m.tau)
                elif m.policy == "no_consolidate":
                    self.assertEqual(m.decision, "keep_both")

    def test_surface_only_sim_is_surface_similarity(self):
        gen = load_generator()
        gloss_sim = {m.pair_id: m.sim for m in self.events if m.policy == "mem0_style"}
        for m in self.events:
            p = self.pmap[m.pair_id]
            with self.subTest(pair=m.pair_id, policy=m.policy):
                if m.policy == "surface_only":
                    self.assertEqual(m.sim, gen.surface_similarity(p.utt_a, p.utt_b))
                else:
                    self.assertEqual(m.sim, gloss_sim[m.pair_id])

    def test_dkmem_v1_gates_only_on_discriminative_features(self):
        discriminative = load_generator().DISCRIMINATIVE_FEATURES
        self.assertEqual(
            discriminative,
            {"kinship", "register", "classifier", "evidentiality", "politeness", "temporal_deixis"},
        )
        self.assertNotIn("name_variant", discriminative)
        for m in self.events:
            if m.policy != "dkmem_v1":
                continue
            p = self.pmap[m.pair_id]
            feature = p.distinction_class.removeprefix("control_")
            a, b = p.distinction_value_a, p.distinction_value_b
            consolidated = m.decision in ("merge", "supersede")
            with self.subTest(pair=m.pair_id, feature=feature):
                if feature in discriminative and a and b and a != b:
                    self.assertEqual(m.decision, "keep_both")
                elif feature in discriminative and (a is None) != (b is None):
                    self.assertEqual(m.decision, "link_unresolved")
                else:
                    self.assertEqual(m.decision, "merge" if m.sim >= m.tau else "keep_both")


if __name__ == "__main__":
    unittest.main()
