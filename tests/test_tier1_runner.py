"""Focused tests for dkmem.tier1.runner. No GPU: the LLM strategy uses a fake
generator. Only small, hand-built Tier1Records are used -- the real 173
sanctioned records are never run through the pipeline here (that's the
deferred full Tier 1 evaluation).

Run: python -m unittest discover -s tests
"""

import json
import unittest
from pathlib import Path

from dkmem.backends.llm import GenerationParams
from dkmem.memory.extract import ExtractionError
from dkmem.memory.lexicon import load_lexicon
from dkmem.tier1.extraction import NO_BACKBONE
from dkmem.tier1.io import PAIRWISE_DECISIONS, Tier1Record
from dkmem.tier1.runner import (
    run_all_episodes_lexicon_llm,
    run_all_episodes_lexicon_only,
    run_episode_lexicon_llm,
    run_episode_lexicon_only,
)

LEXICON_PATH = Path(__file__).parent.parent / "dkmem" / "memory" / "distinction_features.json"


def record(**overrides):
    d = dict(
        eval_pair_id="ep_0001",
        language="hi",
        utterance_a="meri chachi Pune mein rehti hai",
        utterance_b="meri mausi Pune mein rehti hai",
        utterance_a_id="ep_0001_utt_a",
        utterance_b_id="ep_0001_utt_b",
        opaque_entity_id_a="ent_a1",
        opaque_entity_id_b="ent_b1",
    )
    d.update(overrides)
    return Tier1Record(**d)


class FakeGenerator:
    """Answers each prompt with a fixed or utterance-keyed response."""

    def __init__(self, respond, backbone="fake/backbone"):
        self.respond = respond
        self.backbone = backbone
        self.calls = []

    def generate(self, prompts, params=None):
        self.calls.append((prompts, params))
        return [self.respond(p[-1]["content"].removeprefix("Utterance: ")) for p in prompts]


def model_output(utterance, gloss="user did something", distinction=None):
    return json.dumps(
        {"gloss": gloss, "distinction": distinction or {}, "surface": utterance}, ensure_ascii=False
    )


class TestRunEpisodeLexiconOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = load_lexicon(LEXICON_PATH)

    def test_incompatible_kinship_is_no_merge(self):
        rows = run_episode_lexicon_only(record(), self.lex, run_id="run1", tau=0.85)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.decision, "no_merge")
        self.assertEqual(row.compatibility, "incompatible")
        self.assertIsNone(row.predicted_entity_id)
        self.assertEqual(row.strategy, "dk-mem-lexicon")
        self.assertEqual(row.run_id, "run1")
        self.assertEqual(row.threshold, 0.85)

    def test_entry_ids_and_gold_passthrough(self):
        rows = run_episode_lexicon_only(record(), self.lex, run_id="run1")
        row = rows[0]
        self.assertEqual(row.entry_a.entry_id, "ep_0001_utt_a_e0")
        self.assertEqual(row.entry_b.entry_id, "ep_0001_utt_b_e0")
        self.assertEqual(row.entry_a.source_utterance_id, "ep_0001_utt_a")
        self.assertEqual(row.entry_a.gold_entity_id, "ent_a1")
        self.assertEqual(row.entry_b.gold_entity_id, "ent_b1")
        self.assertEqual(row.entry_a.language, "hi")
        self.assertEqual(row.record_id, "ep_0001_utt_a_e0::ep_0001_utt_b_e0")

    def test_identical_utterances_merge(self):
        r = record(utterance_a="Rohan ek botal doodh laaya", utterance_b="Rohan ek botal doodh laaya")
        rows = run_episode_lexicon_only(r, self.lex, run_id="run1", tau=0.85)
        row = rows[0]
        self.assertEqual(row.decision, "merge")
        self.assertEqual(row.compatibility, "compatible")
        self.assertEqual(row.predicted_entity_id, row.entry_a.entry_id)
        self.assertEqual(row.similarity_score, 1.0)

    def test_underdetermined_link(self):
        r = record(utterance_a="meri chachi Pune mein rehti hai", utterance_b="meri aunty Pune mein rehti hai")
        rows = run_episode_lexicon_only(r, self.lex, run_id="run1", tau=0.85)
        row = rows[0]
        self.assertEqual(row.decision, "underdetermined_link")
        self.assertEqual(row.compatibility, "underdetermined")
        self.assertIsNone(row.predicted_entity_id)

    def test_compatibility_is_never_null_for_lexicon_strategy(self):
        for r in (record(), record(utterance_b=record().utterance_a)):
            row = run_episode_lexicon_only(r, self.lex, run_id="run1")[0]
            self.assertIsNotNone(row.compatibility)

    def test_backbone_sentinel_never_leaks_into_output_row(self):
        # NO_BACKBONE is an internal Extraction.backbone sentinel; it must
        # not appear anywhere in the pairwise_eval row (only in the run
        # manifest's backbone field, written separately as null).
        row = run_episode_lexicon_only(record(), self.lex, run_id="run1")[0]
        self.assertNotIn(NO_BACKBONE, row.to_json())

    def test_unsupported_language_yields_no_rows(self):
        r = record(language="not-a-lang!")
        self.assertEqual(run_episode_lexicon_only(r, self.lex, run_id="run1"), [])

    def test_opaque_entity_ids_do_not_affect_decision_or_similarity(self):
        r1 = record(opaque_entity_id_a="ent_x", opaque_entity_id_b="ent_y")
        r2 = record(opaque_entity_id_a="ent_totally_different", opaque_entity_id_b="ent_also_different")
        row1 = run_episode_lexicon_only(r1, self.lex, run_id="run1")[0]
        row2 = run_episode_lexicon_only(r2, self.lex, run_id="run1")[0]
        self.assertEqual(row1.decision, row2.decision)
        self.assertEqual(row1.similarity_score, row2.similarity_score)
        self.assertEqual(row1.compatibility, row2.compatibility)
        # only the passthrough gold_entity_id fields differ
        self.assertNotEqual(row1.entry_a.gold_entity_id, row2.entry_a.gold_entity_id)

    def test_run_all_episodes_independent_and_in_order(self):
        records = [
            record(eval_pair_id="ep_a", utterance_a_id="ep_a_utt_a", utterance_b_id="ep_a_utt_b"),
            record(eval_pair_id="ep_b", utterance_a_id="ep_b_utt_a", utterance_b_id="ep_b_utt_b",
                  utterance_a="Rohan ek botal doodh laaya", utterance_b="Rohan ek botal doodh laaya"),
        ]
        rows = run_all_episodes_lexicon_only(records, self.lex, run_id="run1")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].entry_a.source_utterance_id, "ep_a_utt_a")
        self.assertEqual(rows[1].entry_a.source_utterance_id, "ep_b_utt_a")
        # entry_ids are globally unique across episodes
        ids = [rows[0].entry_a.entry_id, rows[0].entry_b.entry_id,
               rows[1].entry_a.entry_id, rows[1].entry_b.entry_id]
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_state_carries_between_calls(self):
        # Calling twice with the same record gives byte-identical results:
        # nothing (cache, counter, mutable lexicon state) is shared/mutated.
        row1 = run_episode_lexicon_only(record(), self.lex, run_id="run1")[0]
        row2 = run_episode_lexicon_only(record(), self.lex, run_id="run1")[0]
        self.assertEqual(row1, row2)


class TestRunEpisodeLexiconLLM(unittest.TestCase):
    def setUp(self):
        self.lex = load_lexicon(LEXICON_PATH)
        self.params = GenerationParams(seed=0)

    def test_basic_merge_via_fake_llm(self):
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        r = record(utterance_a="Rohan ek botal doodh laaya", utterance_b="Rohan ek botal doodh laaya")
        rows = run_episode_lexicon_llm(r, self.lex, gen, self.params, run_id="run1")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.decision, "merge")
        self.assertEqual(row.strategy, "dk-mem-lexicon-llm")
        self.assertEqual(row.compatibility, "compatible")

    def test_lexicon_overrides_model_guess(self):
        gen = FakeGenerator(lambda u: model_output(u, distinction={"kinship": "chachi"}))
        rows = run_episode_lexicon_llm(record(), self.lex, gen, self.params, run_id="run1")
        row = rows[0]
        # Model wrongly said both sides are "chachi"; the real lexicon
        # corrects side b to "mausi", producing an incompatible pair.
        self.assertEqual(row.decision, "no_merge")
        self.assertEqual(row.compatibility, "incompatible")

    def test_batched_single_generate_call(self):
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        run_episode_lexicon_llm(record(), self.lex, gen, self.params, run_id="run1")
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(len(gen.calls[0][0]), 2)  # both sides in one batch

    def test_malformed_model_output_yields_no_rows(self):
        gen = FakeGenerator(lambda u: "not json")
        rows = run_episode_lexicon_llm(record(), self.lex, gen, self.params, run_id="run1")
        self.assertEqual(rows, [])

    def test_partial_failure_yields_no_rows(self):
        def respond(u):
            return model_output(u) if "chachi" in u else "garbage"

        gen = FakeGenerator(respond)
        rows = run_episode_lexicon_llm(record(), self.lex, gen, self.params, run_id="run1")
        self.assertEqual(rows, [])

    def test_run_all_episodes_llm(self):
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        records = [record(eval_pair_id="ep_a", utterance_a_id="ep_a_utt_a", utterance_b_id="ep_a_utt_b"),
                  record(eval_pair_id="ep_b", utterance_a_id="ep_b_utt_a", utterance_b_id="ep_b_utt_b")]
        rows = run_all_episodes_lexicon_llm(records, self.lex, gen, self.params, run_id="run1")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r.strategy == "dk-mem-lexicon-llm" for r in rows))


class TestDecisionVocabulary(unittest.TestCase):
    def test_all_produced_decisions_are_in_the_external_vocabulary(self):
        lex = load_lexicon(LEXICON_PATH)
        scenarios = [
            record(),  # incompatible
            record(utterance_b=record().utterance_a),  # compatible/merge
            record(utterance_b="meri aunty Pune mein rehti hai"),  # underdetermined
        ]
        for r in scenarios:
            row = run_episode_lexicon_only(r, lex, run_id="run1")[0]
            self.assertIn(row.decision, PAIRWISE_DECISIONS)
            self.assertNotIn(row.decision, ("keep_both", "link_unresolved"))


class TestMemoryEntryContent(unittest.TestCase):
    """gloss/surface/distinction on the output MemoryEntry (pairwise_eval.schema.json)."""

    @classmethod
    def setUpClass(cls):
        cls.lex = load_lexicon(LEXICON_PATH)

    def test_lexicon_only_populates_gloss_surface_distinction(self):
        r = record()  # "meri chachi..." vs "meri mausi..."
        row = run_episode_lexicon_only(r, self.lex, run_id="run1")[0]
        self.assertEqual(row.entry_a.gloss, r.utterance_a)  # no LLM: gloss == raw utterance
        self.assertEqual(row.entry_a.surface, r.utterance_a)
        self.assertEqual(row.entry_a.distinction.cls, "kinship")
        self.assertEqual(row.entry_a.distinction.value, "chachi")
        self.assertEqual(row.entry_a.distinction.extraction_method, "lexicon")
        self.assertEqual(row.entry_b.distinction.value, "mausi")

    def test_lexicon_only_no_marker_gives_null_distinction(self):
        r = record(utterance_a="Rohan ek botal doodh laaya", utterance_b="Rohan ek botal doodh laaya")
        row = run_episode_lexicon_only(r, self.lex, run_id="run1")[0]
        self.assertIsNone(row.entry_a.distinction)
        self.assertIsNone(row.entry_b.distinction)

    def test_register_class_translated_to_honorific_register(self):
        r = record(utterance_a="tum kal aaoge", utterance_b="aap kal aaoge")
        row = run_episode_lexicon_only(r, self.lex, run_id="run1")[0]
        self.assertEqual(row.entry_a.distinction.cls, "honorific_register")
        self.assertEqual(row.entry_a.distinction.value, "tum")

    def test_kinship_class_stays_the_same_name(self):
        row = run_episode_lexicon_only(record(), self.lex, run_id="run1")[0]
        self.assertEqual(row.entry_a.distinction.cls, "kinship")
        self.assertEqual(row.entry_a.distinction.value, "chachi")

    def test_out_of_scope_turkish_evidential_gives_no_distinction(self):
        r = record(language="tr", utterance_a="Ali Ankara'ya gitmiş.", utterance_b="Ali Ankara'ya gitmiş.")
        row = run_episode_lexicon_only(r, self.lex, run_id="run1")[0]
        self.assertIsNone(row.entry_a.distinction)

    def test_llm_strategy_extraction_method_is_lexicon_plus_llm(self):
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        row = run_episode_lexicon_llm(record(), self.lex, gen, GenerationParams(seed=0), run_id="run1")[0]
        self.assertEqual(row.entry_a.distinction.extraction_method, "lexicon+llm")
        self.assertEqual(row.entry_b.distinction.extraction_method, "lexicon+llm")

    def test_llm_strategy_populates_model_gloss_not_raw_utterance(self):
        gen = FakeGenerator(lambda u: model_output(u, gloss="user's aunt lives in Pune"))
        row = run_episode_lexicon_llm(record(), self.lex, gen, GenerationParams(seed=0), run_id="run1")[0]
        self.assertEqual(row.entry_a.gloss, "user's aunt lives in Pune")
        self.assertEqual(row.entry_a.surface, record().utterance_a)

    def test_record_serializes_with_schema_field_names(self):
        row = run_episode_lexicon_only(record(), self.lex, run_id="run1")[0]
        d = json.loads(row.to_json())
        self.assertIn("class", d["entry_a"]["distinction"])
        self.assertNotIn("cls", d["entry_a"]["distinction"])
        self.assertEqual(d["entry_a"]["distinction"]["class"], "kinship")


if __name__ == "__main__":
    unittest.main()
