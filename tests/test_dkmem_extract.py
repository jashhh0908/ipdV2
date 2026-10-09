"""Integration tests for dkmem.memory.dkmem_extract (lexicon + Qwen baseline).

TestRealLexiconIntegration runs the real, pilot-reviewed
dkmem/memory/distinction_features.json through the full ProbeItem ->
baseline extraction -> matcher -> distinction-enhanced Extraction pipeline.
Everything else uses small synthetic Lexicons (built directly, not written
to the JSON file) to exercise scenarios the real 15 entries don't contain by
design -- lexicon ambiguity, custom priorities, etc. No GPU; the baseline
"model" is a fake generator.

Run: python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path

from dkmem.backends.llm import GenerationParams
from dkmem.memory.cache import CacheKey, ExtractionCache
from dkmem.memory.dkmem_extract import (
    DKMEM_LEXICON_SUFFIX,
    apply_lexicon,
    dkmem_extract,
    dkmem_extract_many,
    dkmem_prompt_id,
)
from dkmem.memory.extract import ExtractionBatchError, ExtractionError, extract
from dkmem.memory.lexicon import Lexicon, LexiconEntry, load_lexicon
from dkmem.memory.prompts import DEFAULT_EXTRACTION_PROMPT
from dkmem.memory.schema import ProbeItem

LEXICON_PATH = Path(__file__).parent.parent / "dkmem" / "memory" / "distinction_features.json"


def entry(**overrides):
    d = dict(
        id="e1", lang="hi", distinction_class="register", match_type="exact_word",
        surface_forms=["tu"], value="tu", priority=0, notes="",
    )
    d.update(overrides)
    d["surface_forms"] = tuple(d["surface_forms"])
    return LexiconEntry(**d)


PROBE_TU = ProbeItem(
    pair_id="p1", utt_a="Tu bahut accha khana banata hai.", utt_b="x",
    lang="hi", distinction_class="register", distinction_value_a="tu",
    distinction_value_b=None, gold_same_entity=True, retrieval_query="q",
)
PROBE_MIS = ProbeItem(
    pair_id="p2", utt_a="Ali Ankara'ya gitmiş.", utt_b="x",
    lang="tr", distinction_class="evidentiality", distinction_value_a="reported/hearsay",
    distinction_value_b=None, gold_same_entity=True, retrieval_query="q",
)


class FakeGenerator:
    """Answers each prompt with a fixed or utterance-keyed response."""

    backbone = "fake/backbone"

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def generate(self, prompts, params=None):
        self.calls.append((prompts, params))
        return [self.respond(p[-1]["content"].removeprefix("Utterance: ")) for p in prompts]


def model_output(utterance, gloss="user did something", distinction=None):
    return json.dumps(
        {"gloss": gloss, "distinction": distinction or {}, "surface": utterance},
        ensure_ascii=False,
    )


class TestRealLexiconIntegration(unittest.TestCase):
    """End-to-end: ProbeItem -> baseline extraction -> matcher (the real,
    pilot-reviewed distinction_features.json) -> distinction-enhanced
    Extraction."""

    @classmethod
    def setUpClass(cls):
        cls.lex = load_lexicon(LEXICON_PATH)

    def test_kinship_override_via_real_lexicon(self):
        probe = ProbeItem(
            pair_id="real-1", utt_a="Meri chachi Mumbai mein rehti hai.", utt_b="x",
            lang="hi", distinction_class="kinship", distinction_value_a="chachi",
            distinction_value_b=None, gold_same_entity=True, retrieval_query="q",
        )
        # Baseline (fake Qwen) guesses no distinction at all -- typical of a
        # model that collapses "chachi" to plain "aunt" and drops the marker.
        gen = FakeGenerator(lambda u: model_output(u, gloss="user's aunt lives in Mumbai"))
        result = dkmem_extract(probe, "a", gen, self.lex, params=GenerationParams(seed=0))
        self.assertEqual(result.distinction, {"kinship": "chachi"})
        self.assertEqual(result.prompt_id, DEFAULT_EXTRACTION_PROMPT.prompt_id + DKMEM_LEXICON_SUFFIX)
        self.assertEqual(result.surface, probe.utt_a)

    def test_kinship_override_via_devanagari_utterance(self):
        probe = ProbeItem(
            pair_id="real-2", utt_a="मेरी चाची मुंबई में रहती है।", utt_b="x",
            lang="hi", distinction_class="kinship", distinction_value_a="chachi",
            distinction_value_b=None, gold_same_entity=True, retrieval_query="q",
        )
        gen = FakeGenerator(lambda u: model_output(u, gloss="user's aunt lives in Mumbai"))
        result = dkmem_extract(probe, "a", gen, self.lex, params=GenerationParams(seed=0))
        self.assertEqual(result.distinction, {"kinship": "chachi"})

    def test_register_override_via_real_lexicon(self):
        probe = ProbeItem(
            pair_id="real-3", utt_a="Tum kal office aaoge.", utt_b="x",
            lang="hi", distinction_class="register", distinction_value_a="tum",
            distinction_value_b=None, gold_same_entity=True, retrieval_query="q",
        )
        # Baseline (wrong) guess is overridden by the confident lexicon match.
        gen = FakeGenerator(lambda u: model_output(u, distinction={"register": "aap"}))
        result = dkmem_extract(probe, "a", gen, self.lex, params=GenerationParams(seed=0))
        self.assertEqual(result.distinction, {"register": "tum"})

    def test_out_of_scope_markers_are_not_in_the_real_lexicon(self):
        # The Turkish evidential suffix left the lexicon with the scope change,
        # so the lexicon contributes nothing for this utterance.
        probe = ProbeItem(
            pair_id="real-4", utt_a="Ali Ankara'ya gitmiş.", utt_b="x",
            lang="tr", distinction_class="evidentiality",
            distinction_value_a="reported/hearsay", distinction_value_b=None,
            gold_same_entity=True, retrieval_query="q",
        )
        gen = FakeGenerator(lambda u: model_output(u, gloss="Ali went to Ankara"))
        result = dkmem_extract(probe, "a", gen, self.lex, params=GenerationParams(seed=0))
        self.assertEqual(result.distinction, {})

    def test_unrelated_utterance_gets_no_model_value(self):
        probe = ProbeItem(
            pair_id="real-5", utt_a="Priya ne mera call uthaya.", utt_b="x",
            lang="hi", distinction_class="name_variant", distinction_value_a="Priya-latin",
            distinction_value_b=None, gold_same_entity=True, retrieval_query="q",
        )
        gen = FakeGenerator(
            lambda u: model_output(u, distinction={"name_variant": "Priya-latin"})
        )
        result = dkmem_extract(probe, "a", gen, self.lex, params=GenerationParams(seed=0))
        # No lexicon entry covers name_variant and nothing is ambiguous: the model is not consulted.
        self.assertEqual(result.distinction, {})

    def baseline_for(self, surface, distinction=None):
        return extract(
            ProbeItem("p", surface, surface, "hi", "kinship", None, None, True, "q"),
            "a",
            FakeGenerator(lambda u: model_output(u, distinction=distinction)),
            GenerationParams(seed=0),
        )

    def test_resolve_distinction_needs_the_model_only_for_ambiguity(self):
        from dkmem.memory.dkmem_extract import ambiguous_distinction_classes, resolve_distinction

        self.assertEqual(resolve_distinction(self.lex, "Meri chachi aayi.", "hi", None), {"kinship": "chachi"})
        self.assertEqual(resolve_distinction(self.lex, "Priya ne bola.", "hi", None), {})
        self.assertEqual(ambiguous_distinction_classes(self.lex, "Meri chachi aayi.", "hi"), set())
        # "chachi aur mausi": two different kinship values on one utterance -> ambiguous
        utt = "meri chachi aur mausi aayi."
        self.assertEqual(ambiguous_distinction_classes(self.lex, utt, "hi"), {"kinship"})
        self.assertIsNone(resolve_distinction(self.lex, utt, "hi", None))  # the model's answer is needed and missing
        model = self.baseline_for(utt, distinction={"kinship": "chachi", "register": "tu"})
        self.assertEqual(resolve_distinction(self.lex, utt, "hi", model), {"kinship": "chachi"})  # only the flagged class
        self.assertEqual(resolve_distinction(self.lex, utt, "hi", self.baseline_for(utt, distinction={})), {})

    def test_batch_through_real_lexicon(self):
        probes = [
            ProbeItem("b1", "Meri bua Delhi mein rehti hai.", "x", "hi", "kinship",
                     "bua", None, True, "q"),
            ProbeItem("b2", "Aap kal aaoge.", "x", "hi", "register",
                     "aap", None, True, "q"),
        ]
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        out = dkmem_extract_many(
            [(probes[0], "a"), (probes[1], "a")], gen, self.lex, params=GenerationParams(seed=0)
        )
        self.assertEqual(out[0].distinction, {"kinship": "bua"})
        self.assertEqual(out[1].distinction, {"register": "aap"})
        self.assertEqual(len(gen.calls), 1)  # one batched generate() call


class TestApplyLexicon(unittest.TestCase):
    """Unit-level tests of the merge logic, without going through the model."""

    def setUp(self):
        self.lex = Lexicon((
            entry(id="tu", distinction_class="register", surface_forms=["tu"], value="tu"),
            entry(id="aap", distinction_class="register", match_type="exact_word",
                 surface_forms=["aap"], value="aap"),
        ))

    def baseline(self, surface, distinction=None):
        return extract(
            ProbeItem("p", surface, surface, "hi", "register", None, None, True, "q"),
            "a",
            FakeGenerator(lambda u: model_output(u, distinction=distinction)),
            GenerationParams(seed=0),
        )

    def test_lexicon_fills_unmatched_class(self):
        base = self.baseline("Tu kal aana.", distinction={})
        out = apply_lexicon(base, self.lex, "hi")
        self.assertEqual(out.distinction, {"register": "tu"})

    def test_lexicon_overrides_wrong_model_guess(self):
        base = self.baseline("Tu kal aana.", distinction={"register": "aap"})
        out = apply_lexicon(base, self.lex, "hi")
        self.assertEqual(out.distinction, {"register": "tu"})

    def test_model_value_is_not_used_when_the_lexicon_has_no_opinion(self):
        # Sec 5.3: the model is consulted only for spans the lexicon flags as ambiguous; a class the
        # lexicon does not cover is not tagged even if the model would tag it.
        base = self.baseline("Priya ne bola.", distinction={"name_variant": "Priya-latin", "kinship": "bhaiya"})
        out = apply_lexicon(base, self.lex, "hi")
        self.assertEqual(out.distinction, {})

    def test_model_value_is_ignored_for_a_class_the_lexicon_resolves_unambiguously(self):
        base = self.baseline("Tu kal aana.", distinction={"register": "aap"})
        out = apply_lexicon(base, self.lex, "hi")
        self.assertEqual(out.distinction, {"register": "tu"})

    def test_no_match_and_no_model_value_is_empty(self):
        base = self.baseline("Priya ne bola.", distinction={})
        out = apply_lexicon(base, self.lex, "hi")
        self.assertEqual(out.distinction, {})

    def test_ambiguous_lexicon_defers_to_model(self):
        # Two entries disagree on "register" for the same token "aap":
        # exact_word "aap" -> "formal_aap", prefix "a" -> "vowel_a". This is
        # a real conflict the matcher's own tie-break would resolve, but the
        # integration layer must set it aside and keep the model's answer.
        ambiguous_lex = Lexicon((
            entry(id="x", distinction_class="register", match_type="exact_word",
                 surface_forms=["aap"], value="formal_aap"),
            entry(id="y", distinction_class="register", match_type="prefix",
                 surface_forms=["a"], value="vowel_a"),
        ))
        base = self.baseline("Aap kal aaoge.", distinction={"register": "aap"})
        out = apply_lexicon(base, ambiguous_lex, "hi")
        self.assertEqual(out.distinction, {"register": "aap"})  # model's value kept

    def test_ambiguous_with_no_model_value_omits_class(self):
        ambiguous_lex = Lexicon((
            entry(id="x", distinction_class="register", match_type="exact_word",
                 surface_forms=["aap"], value="formal_aap"),
            entry(id="y", distinction_class="register", match_type="prefix",
                 surface_forms=["a"], value="vowel_a"),
        ))
        base = self.baseline("Aap kal aaoge.", distinction={})
        out = apply_lexicon(base, ambiguous_lex, "hi")
        self.assertEqual(out.distinction, {})

    def test_agreement_across_entries_is_not_ambiguous(self):
        lex = Lexicon((
            entry(id="a", surface_forms=["tu"], value="tu"),
            entry(id="b", match_type="prefix", surface_forms=["t"], value="tu"),
        ))
        base = self.baseline("Tu kal aana.", distinction={})
        out = apply_lexicon(base, lex, "hi")
        self.assertEqual(out.distinction, {"register": "tu"})

    def test_prompt_id_suffixed(self):
        base = self.baseline("Tu kal aana.")
        out = apply_lexicon(base, self.lex, "hi")
        self.assertEqual(base.prompt_id, DEFAULT_EXTRACTION_PROMPT.prompt_id)
        self.assertEqual(out.prompt_id, DEFAULT_EXTRACTION_PROMPT.prompt_id + DKMEM_LEXICON_SUFFIX)
        self.assertEqual(dkmem_prompt_id(DEFAULT_EXTRACTION_PROMPT.prompt_id), out.prompt_id)

    def test_other_fields_copied_verbatim(self):
        base = self.baseline("Tu kal aana.")
        out = apply_lexicon(base, self.lex, "hi")
        for field in ("pair_id", "side", "raw_output", "gloss", "surface",
                      "lang_profile", "backbone", "seed"):
            self.assertEqual(getattr(out, field), getattr(base, field), field)

    def test_does_not_mutate_baseline_extraction(self):
        base = self.baseline("Tu kal aana.", distinction={"register": "aap"})
        original_distinction = dict(base.distinction)
        apply_lexicon(base, self.lex, "hi")
        self.assertEqual(base.distinction, original_distinction)

    def test_code_mixed_lang_checks_both_components(self):
        # "en" has no stub entries; "hi" does. Both are tried without error.
        base = self.baseline("Tu kal aana.", distinction={})
        out = apply_lexicon(base, self.lex, "hi-en")
        self.assertEqual(out.distinction, {"register": "tu"})


class TestBaselineUnchanged(unittest.TestCase):
    """The Mem0/Qwen baseline call itself must be byte-for-byte the same
    whether reached through dkmem_extract or called directly."""

    def test_baseline_extraction_identical_via_both_paths(self):
        lex = Lexicon((entry(),))
        gen_direct = FakeGenerator(lambda u: model_output(u, distinction={}))
        gen_dkmem = FakeGenerator(lambda u: model_output(u, distinction={}))

        direct = extract(PROBE_TU, "a", gen_direct, GenerationParams(seed=0))
        combined = dkmem_extract(PROBE_TU, "a", gen_dkmem, lex, params=GenerationParams(seed=0))

        # Same everything except distinction (lexicon-augmented) and prompt_id.
        for field in ("pair_id", "side", "raw_output", "gloss", "surface",
                      "lang_profile", "backbone", "seed"):
            self.assertEqual(getattr(combined, field), getattr(direct, field), field)
        self.assertEqual(direct.prompt_id, DEFAULT_EXTRACTION_PROMPT.prompt_id)
        self.assertEqual(combined.prompt_id, DEFAULT_EXTRACTION_PROMPT.prompt_id + DKMEM_LEXICON_SUFFIX)


class TestCachePreserved(unittest.TestCase):
    """DK-Mem must reuse the existing baseline cache untouched, never storing
    the lexicon-augmented record in it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache_path = Path(tmp.name) / "extractions.jsonl"
        self.lex = Lexicon((entry(),))
        self.params = GenerationParams(seed=0)

    def test_cache_holds_the_unaugmented_baseline_record(self):
        cache = ExtractionCache(self.cache_path)
        gen = FakeGenerator(lambda u: model_output(u, distinction={"register": "aap"}))
        result = dkmem_extract(PROBE_TU, "a", gen, self.lex, cache, self.params)

        self.assertEqual(len(cache), 1)
        key = CacheKey.build(PROBE_TU, "a", prompt=DEFAULT_EXTRACTION_PROMPT, backbone=gen.backbone,
                             params=self.params)
        cached = cache.get(key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.prompt_id, DEFAULT_EXTRACTION_PROMPT.prompt_id)
        self.assertEqual(cached.distinction, {"register": "aap"})  # raw model output, un-augmented
        self.assertEqual(result.distinction, {"register": "tu"})  # lexicon overrode it
        self.assertEqual(result.prompt_id, DEFAULT_EXTRACTION_PROMPT.prompt_id + DKMEM_LEXICON_SUFFIX)

    def test_second_call_reuses_cache_no_regeneration(self):
        cache = ExtractionCache(self.cache_path)
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        r1 = dkmem_extract(PROBE_TU, "a", gen, self.lex, cache, self.params)
        r2 = dkmem_extract(PROBE_TU, "a", gen, self.lex, cache, self.params)
        self.assertEqual(len(gen.calls), 1)  # only the first call hit the generator
        self.assertEqual(len(cache), 1)  # no extra (lexicon-augmented) entry cached
        self.assertEqual(r1, r2)

    def test_reloaded_cache_still_verifies(self):
        cache = ExtractionCache(self.cache_path)
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        dkmem_extract(PROBE_TU, "a", gen, self.lex, cache, self.params)
        reloaded = ExtractionCache(self.cache_path)  # re-runs full integrity verification
        self.assertEqual(len(reloaded), 1)

    def test_uncached_dkmem_extract_calls_generator_each_time(self):
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        dkmem_extract(PROBE_TU, "a", gen, self.lex, cache=None, params=self.params)
        dkmem_extract(PROBE_TU, "a", gen, self.lex, cache=None, params=self.params)
        self.assertEqual(len(gen.calls), 2)


class TestBatch(unittest.TestCase):
    def setUp(self):
        self.lex = Lexicon((
            entry(id="tu", distinction_class="register", surface_forms=["tu"], value="tu"),
            entry(id="mis", lang="tr", distinction_class="evidentiality", match_type="suffix",
                 surface_forms=["miş"], value="reported/hearsay"),
        ))

    def test_batch_order_and_augmentation(self):
        gen = FakeGenerator(lambda u: model_output(u, distinction={}))
        out = dkmem_extract_many(
            [(PROBE_TU, "a"), (PROBE_MIS, "a")], gen, self.lex, params=GenerationParams(seed=0)
        )
        self.assertEqual(out[0].pair_id, "p1")
        self.assertEqual(out[0].distinction, {"register": "tu"})
        self.assertEqual(out[1].pair_id, "p2")
        self.assertEqual(out[1].distinction, {"evidentiality": "reported/hearsay"})
        self.assertEqual(len(gen.calls), 1)  # one batched generate() call

    def test_empty_batch(self):
        gen = FakeGenerator(lambda u: model_output(u))
        self.assertEqual(dkmem_extract_many([], gen, self.lex), [])
        self.assertEqual(gen.calls, [])

    def test_baseline_batch_failure_propagates_unaugmented(self):
        def respond(u):
            return model_output(u) if u == PROBE_TU.utt_a else "not json"

        gen = FakeGenerator(respond)
        with self.assertRaises(ExtractionBatchError) as ctx:
            dkmem_extract_many([(PROBE_TU, "a"), (PROBE_MIS, "a")], gen, self.lex,
                               params=GenerationParams(seed=0))
        err = ctx.exception
        self.assertEqual(len(err.failures), 1)
        self.assertEqual(err.failures[0].pair_id, "p2")
        # The successful item's Extraction on the exception is the raw
        # baseline record (prompt_id unsuffixed): the lexicon step never ran.
        self.assertEqual(err.extractions[0].prompt_id, DEFAULT_EXTRACTION_PROMPT.prompt_id)
        self.assertIsNone(err.extractions[1])

    def test_single_item_failure_raises_extraction_error(self):
        gen = FakeGenerator(lambda u: "not json")
        with self.assertRaises(ExtractionError):
            dkmem_extract(PROBE_TU, "a", gen, self.lex, params=GenerationParams(seed=0))


if __name__ == "__main__":
    unittest.main()
