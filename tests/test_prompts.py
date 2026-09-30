"""Tests for the frozen extraction prompts in dkmem.memory.prompts.

Run: python -m unittest discover -s tests
"""

import json
import re
import unittest
from pathlib import Path

from dkmem.backends.llm import GenerationParams
from dkmem.memory.extract import DISTINCTION_KEYS, extract, parse_model_output
from dkmem.memory.prompts import (
    MEM0_EXTRACTION_V1,
    MEM0_EXTRACTION_V2,
    MEM0_EXTRACTION_V3,
    PROMPTS,
    get_prompt,
)
from dkmem.memory.schema import ProbeItem, read_jsonl

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Recorded when each prompt was frozen; logged results depend on this text.
FROZEN_SHA256 = {
    "mem0_extraction_v1": "b96c00f2bb5654e01100876801f410cbb1f2b06b26bad21dfd044c3825f340ea",
    "mem0_extraction_v2": "59ebe2b8be62d8bc3b9ac28cb6d5139150f27b399b582325795cdc31dbf02f45",
    "mem0_extraction_v3": "142489d5164e610931b9e143a70ee246912e5162a48b04b87a177e58cafa6aa2",
}
IMPROVED = (MEM0_EXTRACTION_V2, MEM0_EXTRACTION_V3)


def examples(prompt):
    """(utterance, output_json) pairs: chat-turn examples, else inline ones."""
    if prompt.examples:
        return list(prompt.examples)
    lines = prompt.system.splitlines()
    return [
        (line.removeprefix("Utterance: "), lines[i + 1])
        for i, line in enumerate(lines)
        if line.startswith("Utterance: ")
    ]


def full_text(prompt):
    return "\n".join([prompt.system, *(u + "\n" + o for u, o in prompt.examples)])


class FakeGenerator:
    backbone = "fake/backbone"

    def __init__(self, output):
        self.output, self.prompts = output, None

    def generate(self, prompts, params=None):
        self.prompts = prompts
        return [self.output] * len(prompts)


class TestPrompts(unittest.TestCase):
    def test_prompts_are_frozen(self):
        for pid, sha in FROZEN_SHA256.items():
            with self.subTest(prompt=pid):
                self.assertEqual(get_prompt(pid).sha256, sha)

    def test_registry(self):
        self.assertEqual(set(PROMPTS), set(FROZEN_SHA256))
        self.assertEqual(len({p.sha256 for p in PROMPTS.values()}), len(PROMPTS))
        for pid in PROMPTS:
            self.assertIn(pid, DISTINCTION_KEYS)

    def test_same_output_contract(self):
        for p in IMPROVED:
            with self.subTest(prompt=p.prompt_id):
                self.assertEqual(p.output_keys, MEM0_EXTRACTION_V1.output_keys)
                self.assertEqual(p.user_template, MEM0_EXTRACTION_V1.user_template)

    def test_documents_every_allowed_key(self):
        for p in IMPROVED:
            with self.subTest(prompt=p.prompt_id):
                documented = set(re.findall(r'^- "(\w+)":', p.system, re.M))
                self.assertEqual(documented, DISTINCTION_KEYS[p.prompt_id])

    def test_examples_pass_strict_validation(self):
        for p in IMPROVED:
            pairs = examples(p)
            self.assertGreaterEqual(len(pairs), 10, p.prompt_id)
            covered = set()
            for utterance, output in pairs:
                with self.subTest(prompt=p.prompt_id, utterance=utterance):
                    gloss, distinction, _ = parse_model_output(output, utterance=utterance, prompt=p)
                    self.assertFalse(gloss.endswith("."))
                    covered |= set(distinction)
            # Every feature is demonstrated at least once, plus an empty case.
            self.assertEqual(covered, DISTINCTION_KEYS[p.prompt_id], p.prompt_id)
            self.assertIn({}, [json.loads(o)["distinction"] for _, o in pairs])

    def test_v3_renders_examples_as_chat_turns(self):
        messages = MEM0_EXTRACTION_V3.render("target")
        n = len(MEM0_EXTRACTION_V3.examples)
        self.assertEqual(len(messages), 2 + 2 * n)
        self.assertEqual([m["role"] for m in messages],
                         ["system"] + ["user", "assistant"] * n + ["user"])
        self.assertEqual(messages[-1]["content"], "Utterance: target")
        self.assertNotIn("## Examples", MEM0_EXTRACTION_V3.system)

    def test_prompts_without_examples_render_two_messages(self):
        for p in (MEM0_EXTRACTION_V1, MEM0_EXTRACTION_V2):
            self.assertEqual([m["role"] for m in p.render("x")], ["system", "user"])

    def test_no_fixture_leakage(self):
        probes = list(read_jsonl(FIXTURE_DIR / "probe_items.jsonl", ProbeItem))
        for prompt in IMPROVED:
            text = full_text(prompt)
            for p in probes:
                with self.subTest(prompt=prompt.prompt_id, pair=p.pair_id):
                    for utt in (p.utt_a, p.utt_b):
                        self.assertNotIn(utt, text)
                    # Open-vocabulary answers must not appear as quoted values.
                    if p.distinction_class.removeprefix("control_") in ("kinship", "register", "name_variant"):
                        for value in (p.distinction_value_a, p.distinction_value_b):
                            if value:
                                self.assertNotIn(f'"{value}"', text)

    def test_extract_records_prompt_id(self):
        utt = "Meri chachi Mumbai mein rehti hai."
        probe = ProbeItem("x", utt, utt, "hi", "kinship", "chachi", "chachi", True, "q")
        out = json.dumps({"gloss": "user's aunt lives in Mumbai",
                          "distinction": {"kinship": "chachi"}, "surface": utt})
        for p in IMPROVED:
            with self.subTest(prompt=p.prompt_id):
                gen = FakeGenerator(out)
                ex = extract(probe, "a", gen, GenerationParams(seed=1), prompt=p)
                self.assertEqual(ex.prompt_id, p.prompt_id)
                self.assertEqual(gen.prompts, [p.render(utt)])


if __name__ == "__main__":
    unittest.main()
