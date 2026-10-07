"""Tests for dkmem.memory.cache with a fake generator (no GPU, no model).

Run: python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from dkmem.backends.llm import GenerationParams
from dkmem.memory.cache import (
    CacheConflictError,
    CacheError,
    CacheKey,
    ExtractionCache,
    cached_extract,
    cached_extract_many,
)
from dkmem.memory.extract import ExtractionBatchError, ExtractionError, extract
from dkmem.memory.prompts import DEFAULT_EXTRACTION_PROMPT
from dkmem.memory.schema import ProbeItem

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
KIN = {PROBE.utt_a: "chachi", PROBE.utt_b: "mausi"}


def valid_response(utterance, gloss="user's aunt lives in Mumbai"):
    return json.dumps(
        {"gloss": gloss, "distinction": {"kinship": KIN[utterance]}, "surface": utterance},
        ensure_ascii=False,
    )


class FakeGenerator:
    """Answers each prompt from its utterance; counts every generated prompt."""

    def __init__(self, backbone="fake/backbone", respond=valid_response):
        self.backbone = backbone
        self.respond = respond
        self.calls = []

    @property
    def prompts_generated(self):
        return sum(len(p) for p, _ in self.calls)

    def generate(self, prompts, params=None):
        self.calls.append((prompts, params))
        return [self.respond(p[-1]["content"].removeprefix("Utterance: ")) for p in prompts]


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "cache" / "extractions.jsonl"
        self.params = GenerationParams(seed=0)

    def lines(self):
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def write_lines(self, entries):
        self.path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
                             encoding="utf-8")


class TestHitsAndMisses(CacheTestCase):
    def test_first_call_generates_and_persists(self):
        gen, cache = FakeGenerator(), ExtractionCache(self.path)
        ex = cached_extract(PROBE, "a", gen, cache, self.params)
        self.assertEqual(gen.prompts_generated, 1)
        self.assertEqual(len(cache), 1)
        (line,) = self.lines()
        self.assertEqual(line["extraction"], ex.to_dict())

    def test_second_identical_call_does_not_generate(self):
        cached_extract(PROBE, "a", FakeGenerator(), ExtractionCache(self.path), self.params)
        gen = FakeGenerator()
        for cache in (ExtractionCache(self.path),):  # fresh instance: reads from disk
            cached_extract(PROBE, "a", gen, cache, self.params)
            cached_extract(PROBE, "a", gen, cache, self.params)
        self.assertEqual(gen.calls, [])
        self.assertEqual(len(self.lines()), 1)

    def test_different_configuration_does_not_reuse(self):
        cache = ExtractionCache(self.path)
        cached_extract(PROBE, "a", FakeGenerator(), cache, self.params)
        variants = [
            ("seed", FakeGenerator(), GenerationParams(seed=1), None),
            ("max_new_tokens", FakeGenerator(), GenerationParams(seed=0, max_new_tokens=64), None),
            ("sampling", FakeGenerator(), GenerationParams(seed=0, do_sample=True), None),
            ("batch_size", FakeGenerator(), GenerationParams(seed=0, batch_size=1), None),
            ("backbone", FakeGenerator(backbone="other/model"), self.params, None),
            ("backend_info", FakeGenerator(), self.params, {"dtype": "float32"}),
        ]
        for name, gen, params, info in variants:
            with self.subTest(changed=name):
                ex = cached_extract(PROBE, "a", gen, cache, params, backend_info=info)
                self.assertEqual(gen.prompts_generated, 1)
                self.assertEqual(ex.seed, params.seed)
                self.assertEqual(ex.backbone, gen.backbone)
        self.assertEqual(len(cache), 1 + len(variants))
        self.assertEqual(len(self.lines()), 1 + len(variants))

    def test_other_side_is_a_different_entry(self):
        cache = ExtractionCache(self.path)
        a = cached_extract(PROBE, "a", FakeGenerator(), cache, self.params)
        gen = FakeGenerator()
        b = cached_extract(PROBE, "b", gen, cache, self.params)
        self.assertEqual(gen.prompts_generated, 1)
        self.assertEqual((a.side, b.side, b.surface), ("a", "b", PROBE.utt_b))

    def test_cached_record_preserves_raw_output_and_provenance(self):
        raw = "  " + valid_response(PROBE.utt_a) + "\n"
        gen = FakeGenerator(respond=lambda u: raw)
        params = GenerationParams(seed=42)
        original = cached_extract(PROBE, "a", gen, ExtractionCache(self.path), params)
        reloaded = cached_extract(PROBE, "a", FakeGenerator(), ExtractionCache(self.path), params)
        self.assertEqual(reloaded, original)
        self.assertEqual(reloaded.raw_output, raw)
        self.assertEqual((reloaded.pair_id, reloaded.side, reloaded.backbone,
                          reloaded.prompt_id, reloaded.seed),
                         ("t_001", "a", "fake/backbone", DEFAULT_EXTRACTION_PROMPT.prompt_id, 42))
        # Identical to an uncached extraction of the same output.
        self.assertEqual(reloaded, extract(PROBE, "a", FakeGenerator(respond=lambda u: raw), params))

    def test_batch_generates_only_misses_once(self):
        cache = ExtractionCache(self.path)
        cached_extract(PROBE, "a", FakeGenerator(), cache, self.params)
        gen = FakeGenerator()
        out = cached_extract_many([(PROBE, "a"), (PROBE, "b"), (PROBE, "b")], gen, cache, self.params)
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual([p[-1]["content"] for p in gen.calls[0][0]], [f"Utterance: {PROBE.utt_b}"])
        self.assertEqual([e.side for e in out], ["a", "b", "b"])
        self.assertEqual(len(self.lines()), 2)

    def test_batch_failure_keeps_successes_and_caches_nothing_bad(self):
        cache = ExtractionCache(self.path)
        bad = FakeGenerator(respond=lambda u: valid_response(u) if u == PROBE.utt_a else "garbage")
        with self.assertRaises(ExtractionBatchError):
            cached_extract_many([(PROBE, "a"), (PROBE, "b")], bad, cache, self.params)
        self.assertEqual([l["key_fields"]["side"] for l in self.lines()], ["a"])
        with self.assertRaises(ExtractionError):
            cached_extract(PROBE, "b", bad, cache, self.params)
        self.assertEqual(len(ExtractionCache(self.path)), 1)

    def test_key_is_deterministic(self):
        k1 = CacheKey.build(PROBE, "a", prompt=DEFAULT_EXTRACTION_PROMPT, backbone="m", params=self.params,
                            backend_info={"b": 1, "a": 2})
        k2 = CacheKey.build(PROBE, "a", prompt=DEFAULT_EXTRACTION_PROMPT, backbone="m", params=self.params,
                            backend_info={"a": 2, "b": 1})
        self.assertEqual(k1.digest, k2.digest)
        self.assertEqual(len(k1.digest), 64)


class TestNoOverwrite(CacheTestCase):
    def test_conflicting_put_rejected_and_not_written(self):
        cache = ExtractionCache(self.path)
        ex = cached_extract(PROBE, "a", FakeGenerator(), cache, self.params)
        key = CacheKey.build(PROBE, "a", prompt=DEFAULT_EXTRACTION_PROMPT, backbone="fake/backbone",
                             params=self.params)
        other_raw = valid_response(PROBE.utt_a, gloss="user's aunt stays in Mumbai")
        other = replace(ex, raw_output=other_raw, gloss="user's aunt stays in Mumbai")
        with self.assertRaises(CacheConflictError):
            cache.put(key, other)
        self.assertEqual(cache.get(key), ex)
        self.assertEqual(len(self.lines()), 1)

    def test_identical_put_is_noop(self):
        cache = ExtractionCache(self.path)
        ex = cached_extract(PROBE, "a", FakeGenerator(), cache, self.params)
        key = CacheKey.build(PROBE, "a", prompt=DEFAULT_EXTRACTION_PROMPT, backbone="fake/backbone",
                             params=self.params)
        cache.put(key, ex)
        self.assertEqual(len(self.lines()), 1)

    def test_put_rejects_record_not_matching_key(self):
        cache = ExtractionCache(self.path)
        ex = cached_extract(PROBE, "a", FakeGenerator(), cache, self.params)
        key_seed1 = CacheKey.build(PROBE, "a", prompt=DEFAULT_EXTRACTION_PROMPT, backbone="fake/backbone",
                                   params=GenerationParams(seed=1))
        with self.assertRaises(CacheError):
            cache.put(key_seed1, ex)  # record says seed 0


class TestCorruptCache(CacheTestCase):
    def good_entry(self):
        cached_extract(PROBE, "a", FakeGenerator(), ExtractionCache(self.path), self.params)
        (entry,) = self.lines()
        return entry

    def assertLoadFails(self, entries, pattern):
        self.write_lines(entries)
        with self.assertRaisesRegex(CacheError, pattern):
            ExtractionCache(self.path)

    def test_invalid_json(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{"key": \n', encoding="utf-8")
        with self.assertRaisesRegex(CacheError, r"extractions\.jsonl:1: invalid JSON"):
            ExtractionCache(self.path)

    def test_wrong_top_level_shape(self):
        e = self.good_entry()
        self.assertLoadFails([["not", "an", "object"]], "must be an object")
        self.assertLoadFails([{**e, "extra": 1}], "must be an object")

    def test_key_does_not_match_fields(self):
        e = self.good_entry()
        self.assertLoadFails([{**e, "key": "0" * 64}], "does not match the SHA-256")
        fields = {**e["key_fields"], "backbone": "other/model"}  # key left stale
        self.assertLoadFails([{**e, "key_fields": fields}], "does not match the SHA-256")

    def test_missing_key_field(self):
        e = self.good_entry()
        fields = dict(e["key_fields"])
        del fields["lang"]
        self.assertLoadFails([{**e, "key_fields": fields}], "key_fields")

    def test_tampered_extraction(self):
        e = self.good_entry()
        tampered = {**e["extraction"], "gloss": "user's uncle lives in Mumbai"}
        self.assertLoadFails([{**e, "extraction": tampered}], r"differing fields: \['gloss'\]")
        seed_changed = {**e["extraction"], "seed": 9}
        self.assertLoadFails([{**e, "extraction": seed_changed}], r"differing fields: \['seed'\]")

    def test_invalid_extraction_record(self):
        e = self.good_entry()
        broken = dict(e["extraction"])
        del broken["surface"]
        self.assertLoadFails([{**e, "extraction": broken}], "invalid extraction record")

    def test_raw_output_that_no_longer_parses(self):
        e = self.good_entry()
        self.assertLoadFails([{**e, "extraction": {**e["extraction"], "raw_output": "garbage"}}],
                             "does not verify")

    def test_rehashed_forgery_still_caught(self):
        e = self.good_entry()
        fields = {**e["key_fields"], "generation_params": {**e["key_fields"]["generation_params"], "seed": 5}}
        forged = {**e, "key_fields": fields, "key": CacheKey(fields).digest}
        self.assertLoadFails([forged], r"differing fields: \['seed'\]")

    def test_stale_prompt_fingerprint(self):
        e = self.good_entry()
        fields = {**e["key_fields"], "prompt_sha256": "f" * 64}
        self.assertLoadFails([{**e, "key_fields": fields, "key": CacheKey(fields).digest}],
                             "text changed since caching")

    def test_conflicting_duplicate_lines(self):
        e = self.good_entry()
        other_raw = valid_response(PROBE.utt_a, gloss="user's aunt stays in Mumbai")
        other = {**e, "extraction": {**e["extraction"], "raw_output": other_raw,
                                     "gloss": "user's aunt stays in Mumbai"}}
        self.write_lines([e, other])
        with self.assertRaises(CacheConflictError):
            ExtractionCache(self.path)

    def test_identical_duplicate_lines_accepted(self):
        e = self.good_entry()
        self.write_lines([e, e])
        self.assertEqual(len(ExtractionCache(self.path)), 1)


if __name__ == "__main__":
    unittest.main()
