"""Tests for dkmem.memory.extract with a mocked generator (no GPU, no model).

Run: python -m unittest discover -s tests
"""

import json
import unittest
from pathlib import Path

from dkmem.backends.llm import GenerationParams
from dkmem.memory.extract import (
    ExtractionBatchError,
    ExtractionError,
    derive_lang_profile,
    extract,
    extract_many,
    parse_model_output,
)
from dkmem.memory.prompts import (
    DEFAULT_EXTRACTION_PROMPT,
    MEM0_EXTRACTION_V1,
    MEM0_EXTRACTION_V4,
    PromptTemplate,
)
from dkmem.memory.schema import Extraction, ProbeItem, read_jsonl

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FakeGenerator:
    """Returns canned outputs and records what it was called with."""

    def __init__(self, outputs, backbone="fake/backbone"):
        self.outputs = list(outputs)
        self.backbone = backbone
        self.calls = []

    def generate(self, prompts, params=None):
        self.calls.append((prompts, params))
        return self.outputs


PROBE = ProbeItem(
    pair_id="t_001",
    utt_a="Meri chachi Mumbai mein rehti hai.",
    utt_b="Meri mausi Mumbai mein rehti hai.",
    lang="hi",
    distinction_class="kinship",
    distinction_value_a="chachi",
    distinction_value_b="mausi",
    gold_same_entity=False,
    retrieval_query="Meri chachi kahan rehti hai?",
)


def response(gloss="user's aunt lives in Mumbai", distinction=None, surface=PROBE.utt_a):
    return json.dumps(
        {"gloss": gloss, "distinction": {"kinship": "chachi"} if distinction is None else distinction,
         "surface": surface},
        ensure_ascii=False,
    )


class TestExtract(unittest.TestCase):
    def test_builds_extraction_with_provenance(self):
        raw = response()
        gen = FakeGenerator([raw])
        params = GenerationParams(seed=7)
        ex = extract(PROBE, "a", gen, params)
        self.assertIsInstance(ex, Extraction)
        self.assertEqual(ex.pair_id, "t_001")
        self.assertEqual(ex.side, "a")
        self.assertEqual(ex.raw_output, raw)
        self.assertEqual(ex.gloss, "user's aunt lives in Mumbai")
        self.assertEqual(ex.distinction, {"kinship": "chachi"})
        self.assertEqual(ex.surface, PROBE.utt_a)
        self.assertEqual(ex.lang_profile, {"hi": 1.0, "en": 0.0, "script_mix": 0.0})
        self.assertEqual(ex.backbone, "fake/backbone")
        self.assertEqual(ex.prompt_id, DEFAULT_EXTRACTION_PROMPT.prompt_id)
        self.assertEqual(ex.seed, 7)

    def test_default_prompt_is_v4(self):
        self.assertIs(DEFAULT_EXTRACTION_PROMPT, MEM0_EXTRACTION_V4)

    def test_sends_frozen_prompt_and_params(self):
        gen = FakeGenerator([response(surface=PROBE.utt_b, distinction={"kinship": "mausi"})])
        params = GenerationParams(seed=3, max_new_tokens=128)
        extract(PROBE, "b", gen, params)
        (prompts, sent_params), = gen.calls
        self.assertEqual(prompts, [DEFAULT_EXTRACTION_PROMPT.render(PROBE.utt_b)])
        self.assertIs(sent_params, params)

    def test_default_params_seed(self):
        self.assertEqual(extract(PROBE, "a", FakeGenerator([response()])).seed, GenerationParams().seed)

    def test_raw_output_preserved_verbatim(self):
        raw = "  \n" + response() + "\n"
        self.assertEqual(extract(PROBE, "a", FakeGenerator([raw])).raw_output, raw)

    def test_empty_distinction_allowed(self):
        ex = extract(PROBE, "a", FakeGenerator([response(distinction={})]))
        self.assertEqual(ex.distinction, {})

    def test_invalid_side(self):
        with self.assertRaises(ValueError):
            extract(PROBE, "c", FakeGenerator([response()]))

    def test_unregistered_prompt_rejected(self):
        other = PromptTemplate("other_v1", "s", "Utterance: {utterance}", ("gloss", "distinction", "surface"))
        with self.assertRaises(ExtractionError):
            extract(PROBE, "a", FakeGenerator([response()]), prompt=other)


class TestMalformedOutput(unittest.TestCase):
    CASES = {
        "not json": "The user's aunt lives in Mumbai.",
        "code fence": "```json\n" + response() + "\n```",
        "trailing text": response() + "\nHope this helps!",
        "two objects": response() + response(),
        "array": "[" + response() + "]",
        "missing key": json.dumps({"gloss": "g", "distinction": {}}),
        "extra key": json.dumps({"gloss": "g", "distinction": {}, "surface": PROBE.utt_a, "lang": "hi"}),
        "duplicate key": '{"gloss": "a", "gloss": "b", "distinction": {}, "surface": "%s"}' % PROBE.utt_a,
        "empty gloss": response(gloss="  "),
        "non-string gloss": response(gloss=3),
        "distinction not object": response(distinction=["kinship"]),
        "unknown distinction key": response(distinction={"kin": "chachi"}),
        "non-string distinction value": response(distinction={"kinship": 1}),
        "empty distinction value": response(distinction={"kinship": ""}),
        "out-of-scope distinction key": response(distinction={"politeness": "formal"}),
        "surface mismatch": response(surface="Meri chachi Mumbai me rehti hai."),
        "surface not string": response(surface=None),
    }

    def test_each_case_raises_with_context(self):
        for name, raw in self.CASES.items():
            with self.subTest(case=name):
                with self.assertRaises(ExtractionError) as ctx:
                    extract(PROBE, "a", FakeGenerator([raw]))
                err = ctx.exception
                self.assertEqual((err.pair_id, err.side, err.raw_output), ("t_001", "a", raw))
                self.assertIn("t_001/a", str(err))

    def test_closed_values_still_enforced_for_legacy_prompt(self):
        # politeness is a legacy (v1-v3) key with a closed value set.
        raw = response(distinction={"politeness": "polite"})
        with self.assertRaises(ExtractionError):
            extract(PROBE, "a", FakeGenerator([raw]), prompt=MEM0_EXTRACTION_V1)
        ok = response(distinction={"politeness": "formal"})
        ex = extract(PROBE, "a", FakeGenerator([ok]), prompt=MEM0_EXTRACTION_V1)
        self.assertEqual(ex.distinction, {"politeness": "formal"})

    def test_parser_directly(self):
        with self.assertRaises(ExtractionError):
            parse_model_output("{}", utterance=PROBE.utt_a)
        self.assertEqual(
            parse_model_output(response(), utterance=PROBE.utt_a),
            ("user's aunt lives in Mumbai", {"kinship": "chachi"}, PROBE.utt_a),
        )


class TestBatch(unittest.TestCase):
    def test_batch_in_order_one_generate_call(self):
        gen = FakeGenerator([response(), response(surface=PROBE.utt_b, distinction={"kinship": "mausi"})])
        out = extract_many([(PROBE, "a"), (PROBE, "b")], gen, GenerationParams(seed=1))
        self.assertEqual([e.side for e in out], ["a", "b"])
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(len(gen.calls[0][0]), 2)

    def test_failures_reported_with_successes_kept(self):
        good_b = response(surface=PROBE.utt_b, distinction={"kinship": "mausi"})
        gen = FakeGenerator(["garbage", good_b, response(gloss="")])
        with self.assertRaises(ExtractionBatchError) as ctx:
            extract_many([(PROBE, "a"), (PROBE, "b"), (PROBE, "a")], gen)
        err = ctx.exception
        self.assertEqual(len(err.failures), 2)
        self.assertEqual([f.raw_output for f in err.failures], ["garbage", response(gloss="")])
        self.assertIsNone(err.extractions[0])
        self.assertEqual(err.extractions[1].raw_output, good_b)
        self.assertIsNone(err.extractions[2])

    def test_wrong_output_count(self):
        with self.assertRaises(ExtractionError):
            extract_many([(PROBE, "a"), (PROBE, "b")], FakeGenerator([response()]))

    def test_empty(self):
        gen = FakeGenerator([])
        self.assertEqual(extract_many([], gen), [])
        self.assertEqual(gen.calls, [])

    def test_rejects_non_probe(self):
        with self.assertRaises(TypeError):
            extract_many([("t_001", "a")], FakeGenerator([response()]))


class TestLangProfile(unittest.TestCase):
    def test_single_language_tag(self):
        self.assertEqual(derive_lang_profile("hi", "Tum kal office aaoge."),
                         {"hi": 1.0, "en": 0.0, "script_mix": 0.0})
        self.assertEqual(derive_lang_profile("en", "I called her."), {"en": 1.0, "script_mix": 0.0})

    def test_code_mixed_latin(self):
        self.assertEqual(derive_lang_profile("hi-en", "Priya ne mera call uthaya."),
                         {"hi": 0.8, "en": 0.2, "script_mix": 0.0})

    def test_code_mixed_devanagari_with_latin(self):
        self.assertEqual(derive_lang_profile("hi-en", "प्रिया ने मेरा call उठाया."),
                         {"hi": 0.8, "en": 0.2, "script_mix": 0.2})

    def test_japanese_kanji_kana_is_one_script(self):
        self.assertEqual(derive_lang_profile("ja", "水を三杯飲みました。")["script_mix"], 0.0)

    def test_unsupported_tags(self):
        for tag in ("zh-ja", "hi_en", "x", "hi-hi"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                derive_lang_profile(tag, "text")


class TestAgainstFixtures(unittest.TestCase):
    """Feeding each fixture's raw_output back through extraction reproduces it."""

    def test_reproduces_fixture_extractions(self):
        probes = {p.pair_id: p for p in read_jsonl(FIXTURE_DIR / "probe_items.jsonl", ProbeItem)}
        fixtures = list(read_jsonl(FIXTURE_DIR / "extractions.jsonl", Extraction))
        for fx in fixtures:
            with self.subTest(pair=fx.pair_id, side=fx.side):
                gen = FakeGenerator([fx.raw_output], backbone=fx.backbone)
                # The fixtures were written under the frozen v1 prompt and its
                # 7-class key set, so they are replayed through v1.
                ex = extract(probes[fx.pair_id], fx.side, gen, GenerationParams(seed=fx.seed),
                             prompt=MEM0_EXTRACTION_V1)
                self.assertEqual(ex, fx)


if __name__ == "__main__":
    unittest.main()
