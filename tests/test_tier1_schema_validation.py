"""Validates dkmem.tier1 output against Team B's actual JSON schemas, and
runs a small local dry-run producing real run_manifest.json +
pairwise_eval.jsonl files for both in-scope strategies.

This is NOT the full 173 x 3 x 2 evaluation: it uses a 5-record prefix slice
of the real sanctioned input (still hash-verified via load_tier1_input) for
dk-mem-lexicon, and small hand-built records for dk-mem-lexicon-llm (no GPU
available here). No forbidden file is read.

Run: python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from dkmem.backends.llm import GenerationParams
from dkmem.memory.lexicon import load_lexicon
from dkmem.tier1.io import Tier1Record, load_tier1_input, write_pairwise_eval, write_run_manifest
from dkmem.tier1.runner import run_all_episodes_lexicon_llm, run_all_episodes_lexicon_only

REPO_ROOT = Path(__file__).parent.parent
REAL_INPUT_PATH = REPO_ROOT / "dkmem" / "data" / "tier1" / "eval_input" / "teamA_tier1_v2.jsonl"
LEXICON_PATH = REPO_ROOT / "dkmem" / "memory" / "distinction_features.json"
PAIRWISE_SCHEMA_PATH = REPO_ROOT / "dkmem" / "pairwise_eval.schema.json"
RUN_MANIFEST_SCHEMA_PATH = REPO_ROOT / "dkmem" / "run_manifest.schema.json"

DRY_RUN_SIZE = 5  # a slice, never the full 173


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def record(**overrides):
    d = dict(
        eval_pair_id="ep_dryrun01", language="hi",
        utterance_a="meri chachi Pune mein rehti hai", utterance_b="meri mausi Pune mein rehti hai",
        utterance_a_id="ep_dryrun01_utt_a", utterance_b_id="ep_dryrun01_utt_b",
        opaque_entity_id_a="ent_a1", opaque_entity_id_b="ent_b1",
    )
    d.update(overrides)
    return Tier1Record(**d)


class FakeGenerator:
    backbone = "fake/backbone"

    def __init__(self, respond):
        self.respond = respond

    def generate(self, prompts, params=None):
        return [self.respond(p[1]["content"].removeprefix("Utterance: ")) for p in prompts]


def model_output(utterance, gloss="user did something", distinction=None):
    return json.dumps(
        {"gloss": gloss, "distinction": distinction or {}, "surface": utterance}, ensure_ascii=False
    )


class TestPairwiseEvalSchemaValidation(unittest.TestCase):
    """Every row produced by the runner validates against the real schema."""

    @classmethod
    def setUpClass(cls):
        cls.schema = load_schema(PAIRWISE_SCHEMA_PATH)
        cls.lex = load_lexicon(LEXICON_PATH)

    def assertValidRows(self, rows):
        self.assertGreater(len(rows), 0)
        for row in rows:
            jsonschema.validate(row.to_dict(), self.schema)

    def test_incompatible_row_validates(self):
        self.assertValidRows(run_all_episodes_lexicon_only([record()], self.lex, run_id="dry1"))

    def test_merge_row_validates(self):
        rows = run_all_episodes_lexicon_only(
            [record(utterance_b="meri chachi Pune mein rehti hai")], self.lex, run_id="dry1"
        )
        self.assertEqual(rows[0].decision, "merge")
        self.assertValidRows(rows)

    def test_underdetermined_row_validates(self):
        rows = run_all_episodes_lexicon_only(
            [record(utterance_b="meri aunty Pune mein rehti hai")], self.lex, run_id="dry1"
        )
        self.assertEqual(rows[0].decision, "underdetermined_link")
        self.assertValidRows(rows)

    def test_no_distinction_row_validates(self):
        rows = run_all_episodes_lexicon_only(
            [record(utterance_a="Rohan ek botal doodh laaya", utterance_b="Rohan ek gilaas doodh laaya")],
            self.lex, run_id="dry1",
        )
        self.assertIsNone(rows[0].entry_a.distinction)
        self.assertValidRows(rows)

    def test_llm_strategy_row_validates(self):
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        rows = run_all_episodes_lexicon_llm([record()], self.lex, gen, GenerationParams(seed=0), run_id="dry1")
        self.assertValidRows(rows)

    def test_real_input_prefix_all_rows_validate(self):
        # Hash-verifies the real file, then only runs the pipeline on a
        # small prefix -- not the full 173 records.
        records = load_tier1_input(REAL_INPUT_PATH)[:DRY_RUN_SIZE]
        rows = run_all_episodes_lexicon_only(records, self.lex, run_id="dry-real")
        for row in rows:
            jsonschema.validate(row.to_dict(), self.schema)


class TestRunManifestSchemaValidation(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema(RUN_MANIFEST_SCHEMA_PATH)

    def test_lexicon_only_manifest_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_manifest.json"
            write_run_manifest(path, run_id="dry1", strategy="dk-mem-lexicon", seed=0, backbone=None)
            jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), self.schema)

    def test_llm_manifest_with_created_at_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_manifest.json"
            write_run_manifest(
                path, run_id="dry2", strategy="dk-mem-lexicon-llm", seed=1,
                backbone="Qwen/Qwen2.5-3B-Instruct", created_at="2026-09-30T00:00:00Z",
            )
            jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), self.schema)


class TestWorkedExampleFromSpec(unittest.TestCase):
    """experiment_output_spec.md's own worked example must validate against
    the real schema -- a check that this implementation's reading of the
    schema agrees with the spec's own illustration."""

    def test_spec_worked_example_validates(self):
        schema = load_schema(PAIRWISE_SCHEMA_PATH)
        example = {
            "record_id": "run_042_pair_00187", "run_id": "run_042", "strategy": "dk-mem-lexicon",
            "threshold": 0.82, "similarity_score": 0.94, "compatibility": "incompatible",
            "decision": "no_merge", "superseded_entry_id": None, "predicted_entity_id": None,
            "entry_a": {
                "entry_id": "ent_00532", "source_utterance_id": "tier1_probe_0091_utt_a",
                "language": "hi", "surface": "meri chachi Pune mein rehti hai",
                "gloss": "user's aunt lives in Pune",
                "distinction": {"class": "kinship", "value": "chachi", "script": "latn",
                               "extraction_method": "lexicon"},
                "gold_entity_id": "gold_ent_kin_0091_A",
            },
            "entry_b": {
                "entry_id": "ent_00533", "source_utterance_id": "tier1_probe_0091_utt_b",
                "language": "hi", "surface": "meri mausi Pune mein rehti hai",
                "gloss": "user's aunt lives in Pune",
                "distinction": {"class": "kinship", "value": "mausi", "script": "latn",
                               "extraction_method": "lexicon"},
                "gold_entity_id": "gold_ent_kin_0091_B",
            },
        }
        jsonschema.validate(example, schema)


class TestDryRun(unittest.TestCase):
    """A small, complete dry-run: real files written to disk for both
    strategies, then every line validated against the real schemas, plus
    cross-file consistency checks Team B's checklist calls out."""

    @classmethod
    def setUpClass(cls):
        cls.pairwise_schema = load_schema(PAIRWISE_SCHEMA_PATH)
        cls.manifest_schema = load_schema(RUN_MANIFEST_SCHEMA_PATH)
        cls.lex = load_lexicon(LEXICON_PATH)

    def run_dry(self, strategy, rows, run_id, backbone):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_run_manifest(out / "run_manifest.json", run_id=run_id, strategy=strategy,
                              seed=0, backbone=backbone)
            n = write_pairwise_eval(out / "pairwise_eval.jsonl", rows)

            # Exactly two files in the run directory (checklist BLOCK item).
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["pairwise_eval.jsonl", "run_manifest.json"])

            manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
            jsonschema.validate(manifest, self.manifest_schema)

            lines = (out / "pairwise_eval.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), n)
            record_ids = set()
            for line in lines:
                d = json.loads(line)
                jsonschema.validate(d, self.pairwise_schema)
                self.assertEqual(d["run_id"], manifest["run_id"])
                self.assertEqual(d["strategy"], manifest["strategy"])
                self.assertNotIn(d["record_id"], record_ids)  # globally unique within the run
                record_ids.add(d["record_id"])
            return manifest, lines

    def test_dry_run_dk_mem_lexicon_on_real_input_slice(self):
        records = load_tier1_input(REAL_INPUT_PATH)[:DRY_RUN_SIZE]
        rows = run_all_episodes_lexicon_only(records, self.lex, run_id="dryrun_lexicon_seed0")
        manifest, lines = self.run_dry("dk-mem-lexicon", rows, "dryrun_lexicon_seed0", backbone=None)
        self.assertIsNone(manifest["backbone"])
        self.assertLessEqual(len(lines), DRY_RUN_SIZE)  # <=1 comparison per episode today

    def test_dry_run_dk_mem_lexicon_llm(self):
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        records = [record(eval_pair_id=f"ep_dry{i}", utterance_a_id=f"ep_dry{i}_utt_a",
                          utterance_b_id=f"ep_dry{i}_utt_b") for i in range(3)]
        rows = run_all_episodes_lexicon_llm(records, self.lex, gen, GenerationParams(seed=0),
                                            run_id="dryrun_llm_seed0")
        manifest, lines = self.run_dry("dk-mem-lexicon-llm", rows, "dryrun_llm_seed0",
                                       backbone=gen.backbone)
        self.assertEqual(manifest["backbone"], "fake/backbone")
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
