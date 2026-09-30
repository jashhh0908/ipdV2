"""Tests for dkmem.memory.schema. Run: python -m unittest discover -s tests"""

import dataclasses
import os
import tempfile
import unittest

from dkmem.memory.schema import (
    Extraction,
    MergeEvent,
    ProbeItem,
    SchemaError,
    read_jsonl,
    write_jsonl,
)


def probe(**kw):
    d = dict(
        pair_id="kin-001",
        utt_a="meri chachi Pune mein rehti hai",
        utt_b="meri mausi Pune mein rehti hai",
        lang="hi",
        distinction_class="kinship",
        distinction_value_a="chachi",
        distinction_value_b="mausi",
        gold_same_entity=False,
        retrieval_query="meri chachi kahan rehti hai?",
    )
    d.update(kw)
    return ProbeItem(**d)


def extraction(**kw):
    d = dict(
        pair_id="kin-001",
        side="a",
        raw_output='{"gloss": "user\'s aunt lives in Pune"}',
        gloss="user's aunt lives in Pune",
        distinction={"kin": "chachi", "script": "deva"},
        surface="meri chachi Pune mein rehti hai",
        lang_profile={"hi": 0.7, "en": 0.3, "script_mix": 0.4},
        backbone="qwen2.5-1.5b",
        prompt_id="mem0-en",
        seed=0,
    )
    d.update(kw)
    return Extraction(**d)


def merge_event(**kw):
    d = dict(
        pair_id="kin-001",
        policy="dkmem-lexicon",
        tau=0.85,
        sim=0.97,
        decision="keep_both",
        reason="incompatible kin: chachi vs mausi",
        backbone="qwen2.5-1.5b",
        seed=0,
        prompt_id="mem0-en",
    )
    d.update(kw)
    return MergeEvent(**d)


class TestValid(unittest.TestCase):
    def test_construct_and_roundtrip(self):
        for rec in (probe(), extraction(), merge_event()):
            cls = type(rec)
            self.assertEqual(cls.from_json(rec.to_json()), rec)
            self.assertEqual(cls.from_dict(rec.to_dict()), rec)

    def test_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            probe().pair_id = "x"

    def test_unmarked_distinction_value(self):
        self.assertIsNone(probe(distinction_value_b=None).distinction_value_b)

    def test_notes_default(self):
        self.assertEqual(probe().notes, "")

    def test_all_decisions(self):
        for d in ("merge", "supersede", "keep_both", "link_unresolved"):
            self.assertEqual(merge_event(decision=d).decision, d)

    def test_int_tau_coerced_to_float(self):
        self.assertIsInstance(merge_event(tau=1).tau, float)

    def test_non_ascii_preserved_in_json(self):
        rec = probe(utt_a="मेरी चाची पुणे में रहती है")
        self.assertIn("चाची", rec.to_json())

    def test_distinction_is_copied(self):
        src = {"kin": "chachi"}
        rec = extraction(distinction=src)
        src["kin"] = "mausi"
        self.assertEqual(rec.distinction["kin"], "chachi")


class TestInvalid(unittest.TestCase):
    def assertBad(self, fn, **kw):
        with self.assertRaises(SchemaError):
            fn(**kw)

    def test_probe(self):
        self.assertBad(probe, pair_id="")
        self.assertBad(probe, utt_a=None)
        self.assertBad(probe, gold_same_entity=1)
        self.assertBad(probe, gold_same_entity="false")
        self.assertBad(probe, distinction_value_a=3)

    def test_extraction(self):
        self.assertBad(extraction, side="c")
        self.assertBad(extraction, seed="0")
        self.assertBad(extraction, seed=True)
        self.assertBad(extraction, distinction={"kin": 1})
        self.assertBad(extraction, distinction=["kin"])
        self.assertBad(extraction, lang_profile={"hi": "0.7"})
        self.assertBad(extraction, backbone="")

    def test_merge_event(self):
        self.assertBad(merge_event, decision="merged")
        self.assertBad(merge_event, sim="0.9")
        self.assertBad(merge_event, tau=None)
        self.assertBad(merge_event, policy="")

    def test_non_finite_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assertBad(merge_event, tau=bad)
            self.assertBad(merge_event, sim=bad)
            self.assertBad(extraction, lang_profile={"hi": bad})
        self.assertBad(merge_event, sim=10**400)  # overflows float

    def test_non_finite_in_json_rejected(self):
        line = merge_event().to_json().replace('"sim": 0.97', '"sim": NaN')
        with self.assertRaises(SchemaError):
            MergeEvent.from_json(line)

    def test_from_dict_missing_and_unknown(self):
        d = probe().to_dict()
        del d["lang"]
        with self.assertRaises(SchemaError):
            ProbeItem.from_dict(d)
        d = probe().to_dict()
        d["extra"] = 1
        with self.assertRaises(SchemaError):
            ProbeItem.from_dict(d)


class TestJsonl(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "log.jsonl")

    def test_write_read(self):
        recs = [merge_event(), merge_event(pair_id="kin-002", decision="merge")]
        self.assertEqual(write_jsonl(self.path, recs), 2)
        self.assertEqual(list(read_jsonl(self.path, MergeEvent)), recs)

    def test_jsonl_roundtrip_all_types(self):
        cases = [
            (ProbeItem, [probe(utt_a="मेरी चाची पुणे में रहती है", distinction_value_b=None)]),
            (Extraction, [extraction(lang_profile={"hi": 1, "en": 0}), extraction(side="b")]),
            (MergeEvent, [merge_event(tau=1, sim=-0.25)]),
        ]
        for cls, recs in cases:
            write_jsonl(self.path, recs)
            self.assertEqual(list(read_jsonl(self.path, cls)), recs)

    def test_overwrite_is_default(self):
        write_jsonl(self.path, [merge_event(pair_id="old")])
        write_jsonl(self.path, [merge_event(pair_id="new")])
        self.assertEqual([r.pair_id for r in read_jsonl(self.path, MergeEvent)], ["new"])

    def test_append(self):
        write_jsonl(self.path, [merge_event(pair_id="p1")])
        self.assertEqual(write_jsonl(self.path, [merge_event(pair_id="p2")], append=True), 1)
        self.assertEqual(
            [r.pair_id for r in read_jsonl(self.path, MergeEvent)], ["p1", "p2"]
        )

    def test_append_creates_missing_file(self):
        write_jsonl(self.path, [merge_event()], append=True)
        self.assertEqual(len(list(read_jsonl(self.path, MergeEvent))), 1)

    def test_append_after_unterminated_last_line(self):
        with open(self.path, "w", encoding="utf-8", newline="\n") as f:
            f.write(merge_event(pair_id="p1").to_json())  # no trailing newline
        write_jsonl(self.path, [merge_event(pair_id="p2")], append=True)
        self.assertEqual(
            [r.pair_id for r in read_jsonl(self.path, MergeEvent)], ["p1", "p2"]
        )

    def test_read_with_bom(self):
        with open(self.path, "w", encoding="utf-8-sig", newline="\n") as f:
            f.write(probe().to_json() + "\n")
        self.assertEqual(list(read_jsonl(self.path, ProbeItem)), [probe()])

    def test_append_to_bom_file(self):
        with open(self.path, "w", encoding="utf-8-sig", newline="\n") as f:
            f.write(merge_event(pair_id="p1").to_json() + "\n")
        write_jsonl(self.path, [merge_event(pair_id="p2")], append=True)
        self.assertEqual(
            [r.pair_id for r in read_jsonl(self.path, MergeEvent)], ["p1", "p2"]
        )

    def test_read_reports_bad_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(merge_event().to_json() + "\n")
                f.write('{"pair_id": "x"}\n')
            with self.assertRaisesRegex(SchemaError, r"bad\.jsonl:2"):
                list(read_jsonl(path, MergeEvent))


if __name__ == "__main__":
    unittest.main()
