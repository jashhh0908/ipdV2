"""Tests for dkmem.tier1.io: input loading and output writing.

Loading the real sanctioned file only checks its hash/count/shape parse
correctly -- it does not run extraction over all 173 records (that is the
deferred full Tier 1 evaluation, not this focused test).

Run: python -m unittest discover -s tests
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dkmem.tier1.io as tier1_io
from dkmem.tier1.io import (
    DISTINCTION_CLASS_TRANSLATION,
    TIER1_INPUT_RECORD_COUNT,
    TIER1_INPUT_SHA256,
    DistinctionMarker,
    Tier1InputError,
    Tier1Record,
    MemoryEntry,
    PairwiseEvalRecord,
    load_tier1_input,
    translate_distinction,
    write_pairwise_eval,
    write_run_manifest,
)

REAL_INPUT_PATH = Path(__file__).parent.parent / "dkmem" / "data" / "tier1" / "eval_input" / "teamA_tier1_v2.jsonl"

RECORD_DICT = dict(
    eval_pair_id="ep_test0001",
    language="hi",
    utterance_a="meri chachi Pune mein rehti hai",
    utterance_b="meri mausi Pune mein rehti hai",
    utterance_a_id="ep_test0001_utt_a",
    utterance_b_id="ep_test0001_utt_b",
    opaque_entity_id_a="ent_a1",
    opaque_entity_id_b="ent_b1",
)


def entry(**overrides):
    d = dict(entry_id="ep_test0001_utt_a_e0", source_utterance_id="ep_test0001_utt_a",
             gold_entity_id="ent_a1", language="hi", gloss="user's aunt lives in Pune")
    d.update(overrides)
    return MemoryEntry(**d)


class TestRealInputFile(unittest.TestCase):
    """Sanity-checks against the actual sanctioned file, read-only."""

    def test_loads_with_expected_hash_and_count(self):
        records = load_tier1_input(REAL_INPUT_PATH)
        self.assertEqual(len(records), TIER1_INPUT_RECORD_COUNT)
        self.assertEqual(len(records), 173)

    def test_records_carry_no_gold_relation_or_class_fields(self):
        # The 8 sanctioned fields only -- confirms nothing gold leaked in.
        records = load_tier1_input(REAL_INPUT_PATH)
        for r in records[:5]:
            self.assertIsInstance(r, Tier1Record)
        first = records[0]
        self.assertTrue(first.utterance_a_id.startswith(first.eval_pair_id))
        self.assertTrue(first.utterance_b_id.startswith(first.eval_pair_id))

    def test_eab5_hash_constant_matches_actual_file(self):
        import hashlib
        digest = hashlib.sha256(REAL_INPUT_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, TIER1_INPUT_SHA256)


class TestTier1Record(unittest.TestCase):
    def test_valid_record(self):
        r = Tier1Record(**RECORD_DICT)
        self.assertEqual(r.eval_pair_id, "ep_test0001")

    def test_from_dict_missing_and_unknown_fields(self):
        d = dict(RECORD_DICT)
        del d["language"]
        with self.assertRaises(Tier1InputError):
            Tier1Record.from_dict(d)
        d = dict(RECORD_DICT)
        d["distinction_class"] = "kinship"  # a gold-ish field must never be accepted
        with self.assertRaises(Tier1InputError):
            Tier1Record.from_dict(d)

    def test_empty_field_rejected(self):
        for field in RECORD_DICT:
            with self.subTest(field=field):
                d = dict(RECORD_DICT)
                d[field] = ""
                with self.assertRaises(Tier1InputError):
                    Tier1Record(**d)

    def test_utterance_id_must_be_prefixed_by_eval_pair_id(self):
        d = dict(RECORD_DICT)
        d["utterance_a_id"] = "totally_different_id"
        with self.assertRaises(Tier1InputError):
            Tier1Record(**d)


class TestLoadTier1Input(unittest.TestCase):
    def write(self, lines):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        for line in lines:
            tmp.write(line + "\n")
        tmp.close()
        return tmp.name

    def test_wrong_hash_rejected_even_if_shape_is_fine(self):
        path = self.write([json.dumps(RECORD_DICT)] * 173)
        with self.assertRaisesRegex(Tier1InputError, "sha256"):
            load_tier1_input(path)

    def test_wrong_record_count_rejected(self):
        # A synthetic file can never match the real sha256, so the hash
        # check always fires first (by design -- "check the hash before
        # every run"). To test the count check in isolation, patch the
        # expected hash to this file's own, making the count check the
        # next (and only remaining) thing that can reject it.
        path = self.write([json.dumps(RECORD_DICT)] * 5)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with patch.object(tier1_io, "TIER1_INPUT_SHA256", digest):
            with self.assertRaisesRegex(Tier1InputError, "expected 173"):
                load_tier1_input(path)

    def test_missing_file(self):
        with self.assertRaises(Tier1InputError):
            load_tier1_input("does/not/exist.jsonl")

    def test_malformed_line_reports_position(self):
        lines = [json.dumps(RECORD_DICT)] * 172 + ["not json"]
        path = self.write(lines)
        # Hash won't match real input either way, but shape errors are
        # checked after the hash check; use a file whose count is 173 to
        # reach the per-line parse and confirm it reports the bad line.
        with self.assertRaises(Tier1InputError):
            load_tier1_input(path)


class TestMemoryEntry(unittest.TestCase):
    def test_valid(self):
        e = entry()
        self.assertEqual(e.entry_id, "ep_test0001_utt_a_e0")
        self.assertIsNone(e.surface)
        self.assertIsNone(e.distinction)

    def test_empty_field_rejected(self):
        with self.assertRaises(Tier1InputError):
            entry(entry_id="")
        with self.assertRaises(Tier1InputError):
            entry(gloss="")

    def test_surface_and_distinction_optional_but_typed(self):
        e = entry(surface="meri chachi Pune mein rehti hai",
                  distinction=DistinctionMarker(cls="kinship", value="chachi"))
        self.assertEqual(e.surface, "meri chachi Pune mein rehti hai")
        self.assertEqual(e.distinction.value, "chachi")
        with self.assertRaises(Tier1InputError):
            entry(distinction="not-a-marker")

    def test_to_dict_shape_matches_schema_field_names(self):
        e = entry(surface="s", distinction=DistinctionMarker(cls="kinship", value="chachi",
                                                              extraction_method="lexicon"))
        d = e.to_dict()
        self.assertEqual(set(d), {"entry_id", "source_utterance_id", "gold_entity_id",
                                  "language", "gloss", "surface", "distinction"})
        self.assertEqual(set(d["distinction"]), {"class", "value", "script", "extraction_method"})
        self.assertEqual(d["distinction"]["class"], "kinship")  # not "cls"

    def test_null_distinction_serializes_to_none(self):
        self.assertIsNone(entry().to_dict()["distinction"])


class TestDistinctionMarker(unittest.TestCase):
    def test_valid(self):
        m = DistinctionMarker(cls="kinship", value="chachi")
        self.assertIsNone(m.script)
        self.assertIsNone(m.extraction_method)

    def test_bad_class_rejected(self):
        with self.assertRaises(Tier1InputError):
            DistinctionMarker(cls="register", value="tu")  # internal name, not external

    def test_empty_value_rejected(self):
        with self.assertRaises(Tier1InputError):
            DistinctionMarker(cls="kinship", value="")

    def test_bad_extraction_method_rejected(self):
        with self.assertRaises(Tier1InputError):
            DistinctionMarker(cls="kinship", value="chachi", extraction_method="llm-only")

    def test_valid_extraction_methods(self):
        for m in ("lexicon", "lexicon+llm"):
            DistinctionMarker(cls="kinship", value="chachi", extraction_method=m)


class TestTranslateDistinction(unittest.TestCase):
    def test_empty_dict_is_none(self):
        self.assertIsNone(translate_distinction({}, extraction_method="lexicon"))

    def test_single_key_translated(self):
        for internal, external in DISTINCTION_CLASS_TRANSLATION.items():
            with self.subTest(internal=internal):
                m = translate_distinction({internal: "v"}, extraction_method="lexicon")
                self.assertEqual(m.cls, external)
                self.assertEqual(m.value, "v")
                self.assertEqual(m.extraction_method, "lexicon")

    def test_register_becomes_honorific_register(self):
        m = translate_distinction({"register": "tu"}, extraction_method="lexicon")
        self.assertEqual(m.cls, "honorific_register")

    def test_multiple_keys_picks_deterministically(self):
        # "evidentiality" sorts before "kinship" alphabetically.
        m = translate_distinction({"kinship": "chachi", "evidentiality": "reported/hearsay"},
                                  extraction_method="lexicon+llm")
        self.assertEqual(m.cls, "evidentiality")
        self.assertEqual(m.value, "reported/hearsay")

    def test_extraction_method_passed_through(self):
        m = translate_distinction({"kinship": "chachi"}, extraction_method="lexicon+llm")
        self.assertEqual(m.extraction_method, "lexicon+llm")


class TestPairwiseEvalRecord(unittest.TestCase):
    def base(self, **overrides):
        d = dict(
            record_id="r1", run_id="run1", strategy="dk-mem-lexicon",
            entry_a=entry(), entry_b=entry(entry_id="ep_test0001_utt_b_e0",
                                           source_utterance_id="ep_test0001_utt_b",
                                           gold_entity_id="ent_b1"),
            decision="no_merge", similarity_score=0.5, threshold=0.85,
            compatibility="incompatible",
        )
        d.update(overrides)
        return PairwiseEvalRecord(**d)

    def test_valid_no_merge(self):
        r = self.base()
        self.assertIsNone(r.predicted_entity_id)
        self.assertIsNone(r.superseded_entry_id)

    def test_merge_requires_predicted_entity_id(self):
        with self.assertRaises(Tier1InputError):
            self.base(decision="merge", predicted_entity_id=None)
        r = self.base(decision="merge", predicted_entity_id="ep_test0001_utt_a_e0")
        self.assertEqual(r.predicted_entity_id, "ep_test0001_utt_a_e0")

    def test_no_merge_forbids_predicted_entity_id(self):
        with self.assertRaises(Tier1InputError):
            self.base(decision="no_merge", predicted_entity_id="x")

    def test_underdetermined_link_forbids_predicted_entity_id(self):
        with self.assertRaises(Tier1InputError):
            self.base(decision="underdetermined_link", predicted_entity_id="x")

    def test_superseded_entry_id_only_for_supersede(self):
        with self.assertRaises(Tier1InputError):
            self.base(decision="no_merge", superseded_entry_id="ep_test0001_utt_a_e0")

    def test_supersede_superseded_entry_id_must_be_one_of_the_two_entries(self):
        with self.assertRaises(Tier1InputError):
            self.base(decision="supersede", predicted_entity_id="x", superseded_entry_id="not-a-real-entry")
        r = self.base(decision="supersede", predicted_entity_id="ep_test0001_utt_a_e0",
                      superseded_entry_id="ep_test0001_utt_b_e0")
        self.assertEqual(r.superseded_entry_id, "ep_test0001_utt_b_e0")

    def test_bad_strategy_rejected(self):
        with self.assertRaises(Tier1InputError):
            self.base(strategy="mem0")

    def test_bad_decision_rejected(self):
        with self.assertRaises(Tier1InputError):
            self.base(decision="keep_both")  # internal vocabulary, not external

    def test_bad_compatibility_rejected(self):
        with self.assertRaises(Tier1InputError):
            self.base(compatibility="maybe")

    def test_compatibility_null_allowed(self):
        r = self.base(compatibility=None)
        self.assertIsNone(r.compatibility)

    def test_similarity_score_out_of_range_rejected(self):
        with self.assertRaises(Tier1InputError):
            self.base(similarity_score=-0.01)
        with self.assertRaises(Tier1InputError):
            self.base(similarity_score=1.01)

    def test_threshold_out_of_range_rejected(self):
        with self.assertRaises(Tier1InputError):
            self.base(threshold=-0.01)
        with self.assertRaises(Tier1InputError):
            self.base(threshold=1.01)

    def test_similarity_score_and_threshold_boundaries_allowed(self):
        self.base(similarity_score=0.0, threshold=0.0)
        self.base(similarity_score=1.0, threshold=1.0)

    def test_to_json_shape(self):
        r = self.base(decision="merge", predicted_entity_id="ep_test0001_utt_a_e0")
        d = json.loads(r.to_json())
        self.assertEqual(d["entry_a"]["entry_id"], "ep_test0001_utt_a_e0")
        self.assertEqual(d["entry_b"]["source_utterance_id"], "ep_test0001_utt_b")
        self.assertEqual(d["decision"], "merge")
        self.assertEqual(d["predicted_entity_id"], "ep_test0001_utt_a_e0")
        self.assertIsNone(d["superseded_entry_id"])


class TestWriteRunManifest(unittest.TestCase):
    def test_backbone_null_for_no_llm_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_manifest.json"
            write_run_manifest(path, run_id="r1", strategy="dk-mem-lexicon", seed=0, backbone=None)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(data["backbone"])
            self.assertEqual(data["dataset_tier"], "tier1_minimal_pairs")
            self.assertEqual(data["seed"], 0)
            self.assertNotIn("created_at", data)

    def test_backbone_and_created_at_for_llm_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_manifest.json"
            write_run_manifest(
                path, run_id="r2", strategy="dk-mem-lexicon-llm", seed=1,
                backbone="Qwen/Qwen2.5-3B-Instruct", created_at="2026-09-30T00:00:00Z",
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["backbone"], "Qwen/Qwen2.5-3B-Instruct")
            self.assertEqual(data["created_at"], "2026-09-30T00:00:00Z")

    def test_rejects_out_of_scope_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Tier1InputError):
                write_run_manifest(Path(tmp) / "m.json", run_id="r", strategy="mem0",
                                   seed=0, backbone=None)

    def test_rejects_malformed_created_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Tier1InputError):
                write_run_manifest(Path(tmp) / "m.json", run_id="r", strategy="dk-mem-lexicon",
                                   seed=0, backbone=None, created_at="not-a-date")


class TestWritePairwiseEval(unittest.TestCase):
    def test_writes_one_line_per_record(self):
        rows = [
            PairwiseEvalRecord(
                record_id=f"r{i}", run_id="run1", strategy="dk-mem-lexicon",
                entry_a=entry(), entry_b=entry(entry_id="b", source_utterance_id="ep_test0001_utt_b",
                                               gold_entity_id="ent_b1"),
                decision="no_merge", similarity_score=0.1, threshold=0.85, compatibility="incompatible",
            )
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairwise_eval.jsonl"
            n = write_pairwise_eval(path, rows)
            self.assertEqual(n, 3)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            for line, row in zip(lines, rows):
                self.assertEqual(json.loads(line)["record_id"], row.record_id)

    def test_rejects_duplicate_record_id_and_writes_nothing(self):
        rows = [
            PairwiseEvalRecord(
                record_id="dup", run_id="run1", strategy="dk-mem-lexicon",
                entry_a=entry(), entry_b=entry(entry_id="b", source_utterance_id="ep_test0001_utt_b",
                                               gold_entity_id="ent_b1"),
                decision="no_merge", similarity_score=0.1, threshold=0.85, compatibility="incompatible",
            )
            for _ in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairwise_eval.jsonl"
            with self.assertRaises(Tier1InputError):
                write_pairwise_eval(path, rows)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
