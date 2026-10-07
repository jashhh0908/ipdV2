"""Tests for dkmem.config (stage-attribution configs A-D, DK-Mem modes,
strategy names) and dkmem.memory.scope (which distinction classes are in scope).

Run: python -m unittest discover -s tests
"""

import unittest

from dkmem.config import (
    DKMEM_MODES,
    PIPELINE_CONFIG_IDS,
    PIPELINE_CONFIGS,
    STRATEGIES,
    PipelineConfig,
    dkmem_gate_enabled,
    dkmem_mode_for_strategy,
    get_pipeline_config,
    validate_dkmem_mode,
)
from dkmem.memory.extract import DISTINCTION_KEYS
from dkmem.memory.gate import DEFAULT_DISCRIMINATIVE_FEATURES
from dkmem.memory.prompts import DEFAULT_EXTRACTION_PROMPT
from dkmem.memory.scope import (
    DISCRIMINATIVE_CLASSES,
    IN_SCOPE_CLASSES,
    LEGACY_CLASSES,
    LEGACY_DISCRIMINATIVE_CLASSES,
    OUT_OF_SCOPE_CLASSES,
)
from dkmem.tier1.io import TIER1_STRATEGIES


class TestPipelineConfigs(unittest.TestCase):
    def test_four_configs_a_to_d(self):
        self.assertEqual(PIPELINE_CONFIG_IDS, ("A", "B", "C", "D"))
        self.assertEqual(tuple(PIPELINE_CONFIGS), PIPELINE_CONFIG_IDS)

    def test_table_matches_the_research_plan(self):
        # DKMEM_NEW_RESEARCH_IDEA.md Sec 6.2.
        expected = {
            "A": ("english_forced_prompt", "english_gloss", "llm_judge", "L1"),
            "B": ("native_language_prompt", "extractor_output", "llm_judge", "L2"),
            "C": ("none_verbatim", "surface_text", "llm_judge", "L3"),
            "D": ("none_verbatim", "surface_text", "embedding_threshold", "L4"),
        }
        for config_id, row in expected.items():
            with self.subTest(config=config_id):
                c = get_pipeline_config(config_id)
                self.assertEqual(
                    (c.extraction, c.storage, c.merge_mechanism, c.isolates), row
                )
                self.assertEqual(c.config_id, config_id)

    def test_each_config_isolates_a_different_loss_stage(self):
        self.assertEqual([c.isolates for c in PIPELINE_CONFIGS.values()], ["L1", "L2", "L3", "L4"])

    def test_only_config_d_uses_the_embedding_threshold(self):
        self.assertEqual(
            [i for i, c in PIPELINE_CONFIGS.items() if c.merge_mechanism == "embedding_threshold"], ["D"]
        )

    def test_verbatim_configs_do_not_extract(self):
        for i, c in PIPELINE_CONFIGS.items():
            self.assertEqual(c.extraction == "none_verbatim", c.storage == "surface_text", i)

    def test_unknown_config_id(self):
        with self.assertRaises(KeyError):
            get_pipeline_config("E")

    def test_invalid_fields_rejected(self):
        good = dict(config_id="A", extraction="english_forced_prompt", storage="english_gloss",
                    merge_mechanism="llm_judge", isolates="L1")
        PipelineConfig(**good)
        for field, bad in (("config_id", "E"), ("extraction", "x"), ("storage", "x"),
                           ("merge_mechanism", "x"), ("isolates", "L5")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                PipelineConfig(**{**good, field: bad})

    def test_configs_are_immutable(self):
        with self.assertRaises(Exception):
            get_pipeline_config("A").isolates = "L2"
        with self.assertRaises(TypeError):
            PIPELINE_CONFIGS["E"] = get_pipeline_config("A")


class TestDkmemModes(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(DKMEM_MODES, ("off", "lexicon", "lexicon+llm"))

    def test_gate_enabled_unless_off(self):
        self.assertFalse(dkmem_gate_enabled("off"))
        self.assertTrue(dkmem_gate_enabled("lexicon"))
        self.assertTrue(dkmem_gate_enabled("lexicon+llm"))

    def test_validate_rejects_unknown(self):
        for bad in ("on", "OFF", "", None, "lexicon-llm"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_dkmem_mode(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dkmem_gate_enabled(bad)
        self.assertEqual(validate_dkmem_mode("lexicon"), "lexicon")


class TestStrategies(unittest.TestCase):
    def test_unique_and_complete(self):
        self.assertEqual(len(STRATEGIES), len(set(STRATEGIES)))
        for name in ("mem0", "a-mem", "store-surface-only", "embedding-threshold", "raise-tau",
                     "prompt-informed-judge", "flat-dense-rag", "dk-mem-lexicon", "dk-mem-lexicon-llm"):
            self.assertIn(name, STRATEGIES)

    def test_dropped_baselines_are_gone(self):
        for name in ("lightmem", "trag", "crossrag"):
            self.assertNotIn(name, STRATEGIES)

    def test_implemented_strategies_are_a_subset(self):
        self.assertLessEqual(set(TIER1_STRATEGIES), set(STRATEGIES))

    def test_dkmem_mode_implied_by_strategy(self):
        self.assertEqual(dkmem_mode_for_strategy("dk-mem-lexicon"), "lexicon")
        self.assertEqual(dkmem_mode_for_strategy("dk-mem-lexicon-llm"), "lexicon+llm")
        for name in STRATEGIES:
            if not name.startswith("dk-mem"):
                self.assertEqual(dkmem_mode_for_strategy(name), "off", name)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            dkmem_mode_for_strategy("lightmem")


class TestScope(unittest.TestCase):
    def test_in_scope_classes(self):
        self.assertEqual(IN_SCOPE_CLASSES, {"kinship", "register", "name_variant"})

    def test_cannot_link_classes_are_a_subset_and_exclude_name_variant(self):
        self.assertEqual(DISCRIMINATIVE_CLASSES, {"kinship", "register"})
        self.assertLessEqual(DISCRIMINATIVE_CLASSES, IN_SCOPE_CLASSES)
        self.assertNotIn("name_variant", DISCRIMINATIVE_CLASSES)

    def test_out_of_scope_classes(self):
        self.assertEqual(OUT_OF_SCOPE_CLASSES, {"politeness", "evidentiality", "classifier", "temporal_deixis"})
        self.assertFalse(OUT_OF_SCOPE_CLASSES & IN_SCOPE_CLASSES)

    def test_legacy_sets_are_the_pre_rescope_ones(self):
        self.assertEqual(LEGACY_CLASSES, IN_SCOPE_CLASSES | OUT_OF_SCOPE_CLASSES)
        self.assertEqual(
            LEGACY_DISCRIMINATIVE_CLASSES,
            {"kinship", "register", "classifier", "evidentiality", "politeness", "temporal_deixis"},
        )

    def test_pipeline_defaults_follow_the_scope(self):
        self.assertEqual(DEFAULT_DISCRIMINATIVE_FEATURES, DISCRIMINATIVE_CLASSES)
        self.assertEqual(DISTINCTION_KEYS[DEFAULT_EXTRACTION_PROMPT.prompt_id], IN_SCOPE_CLASSES)


if __name__ == "__main__":
    unittest.main()
