"""Tests for the frozen extraction prompts in dkmem.memory.prompts.

Run: python -m unittest discover -s tests
"""

import json
import re
import unittest
from pathlib import Path

from dkmem.backends.llm import GenerationParams
from dkmem.memory.extract import DISTINCTION_KEYS, ExtractionError, extract, parse_model_output
from dkmem.memory.prompts import (
    DEFAULT_EXTRACTION_PROMPT,
    MEM0_EXTRACTION_V1,
    MEM0_EXTRACTION_V2,
    MEM0_EXTRACTION_V3,
    MEM0_EXTRACTION_V4,
    PROMPTS,
    get_prompt,
)
from dkmem.memory.schema import ProbeItem, read_jsonl
from dkmem.memory.scope import IN_SCOPE_CLASSES, LEGACY_CLASSES, OUT_OF_SCOPE_CLASSES

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Recorded when each prompt was frozen; logged results depend on this text.
FROZEN_SHA256 = {
    "mem0_extraction_v1": "b96c00f2bb5654e01100876801f410cbb1f2b06b26bad21dfd044c3825f340ea",
    "mem0_extraction_v2": "59ebe2b8be62d8bc3b9ac28cb6d5139150f27b399b582325795cdc31dbf02f45",
    "mem0_extraction_v3": "142489d5164e610931b9e143a70ee246912e5162a48b04b87a177e58cafa6aa2",
    "mem0_extraction_v4": "3b4e1de73ac89d5142d0e0cfba52a18e03307f87ac69b2646fb1c9abbabdf00a",
}
IMPROVED = (MEM0_EXTRACTION_V2, MEM0_EXTRACTION_V3, MEM0_EXTRACTION_V4)
CHAT_TURN_PROMPTS = (MEM0_EXTRACTION_V3, MEM0_EXTRACTION_V4)


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
            self.assertGreaterEqual(len(pairs), 9, p.prompt_id)
            covered = set()
            for utterance, output in pairs:
                with self.subTest(prompt=p.prompt_id, utterance=utterance):
                    gloss, distinction, _ = parse_model_output(output, utterance=utterance, prompt=p)
                    self.assertFalse(gloss.endswith("."))
                    covered |= set(distinction)
            # Every feature is demonstrated at least once, plus an empty case.
            self.assertEqual(covered, DISTINCTION_KEYS[p.prompt_id], p.prompt_id)
            self.assertIn({}, [json.loads(o)["distinction"] for _, o in pairs])

    def test_v3_and_v4_render_examples_as_chat_turns(self):
        for p in CHAT_TURN_PROMPTS:
            with self.subTest(prompt=p.prompt_id):
                messages = p.render("target")
                n = len(p.examples)
                self.assertEqual(len(messages), 2 + 2 * n)
                self.assertEqual([m["role"] for m in messages],
                                 ["system"] + ["user", "assistant"] * n + ["user"])
                # The real utterance is always the last message, not messages[1].
                self.assertEqual(messages[-1]["content"], "Utterance: target")
                self.assertNotIn("## Examples", p.system)

    def test_legacy_prompts_keep_the_old_seven_class_key_set(self):
        for p in (MEM0_EXTRACTION_V1, MEM0_EXTRACTION_V2, MEM0_EXTRACTION_V3):
            self.assertEqual(DISTINCTION_KEYS[p.prompt_id], LEGACY_CLASSES)
        self.assertEqual(len(LEGACY_CLASSES), 7)

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


class TestFinalPromptV4(unittest.TestCase):
    """v4 is the final prompt: v3 re-scoped to kinship / register / name_variant."""

    FULL = MEM0_EXTRACTION_V4.system + "\n" + "\n".join(
        u + "\n" + o for u, o in MEM0_EXTRACTION_V4.examples
    )

    def test_is_the_default_extraction_prompt(self):
        self.assertIs(DEFAULT_EXTRACTION_PROMPT, MEM0_EXTRACTION_V4)
        self.assertIs(get_prompt("mem0_extraction_v4"), MEM0_EXTRACTION_V4)

    def test_allowed_keys_are_exactly_the_in_scope_classes(self):
        self.assertEqual(DISTINCTION_KEYS["mem0_extraction_v4"], IN_SCOPE_CLASSES)
        self.assertEqual(IN_SCOPE_CLASSES, {"kinship", "register", "name_variant"})

    def test_prompt_never_mentions_an_out_of_scope_class(self):
        for term in OUT_OF_SCOPE_CLASSES:
            with self.subTest(term=term):
                self.assertNotIn(term, self.FULL)

    def test_out_of_scope_keys_are_rejected_by_the_parser(self):
        utt = "Maine kal pizza khaya."
        for key, value in (("temporal_deixis", "yesterday"), ("politeness", "formal"),
                           ("evidentiality", "reported/hearsay"), ("classifier", "cup")):
            raw = json.dumps({"gloss": "user ate pizza yesterday", "distinction": {key: value},
                              "surface": utt})
            with self.subTest(key=key), self.assertRaisesRegex(ExtractionError, "unknown keys"):
                parse_model_output(raw, utterance=utt, prompt=MEM0_EXTRACTION_V4)

    def test_in_scope_keys_are_accepted(self):
        utt = "Meri nani Pune mein rehti hai."
        for distinction in ({"kinship": "nani"}, {"register": "tum"},
                            {"name_variant": "Priya-latin"}, {}):
            raw = json.dumps({"gloss": "user's grandmother lives in Pune",
                              "distinction": distinction, "surface": utt})
            with self.subTest(distinction=distinction):
                parse_model_output(raw, utterance=utt, prompt=MEM0_EXTRACTION_V4)

    def test_gloss_and_surface_rules_are_the_same_as_v3(self):
        # v4 changes only the distinction section and the examples.
        def section(text, start, end):
            return text[text.index(start):text.index(end)]

        self.assertEqual(section(MEM0_EXTRACTION_V3.system, "## gloss", "## distinction"),
                         section(MEM0_EXTRACTION_V4.system, "## gloss", "## distinction"))
        self.assertEqual(MEM0_EXTRACTION_V3.system[MEM0_EXTRACTION_V3.system.index("## surface"):],
                         MEM0_EXTRACTION_V4.system[MEM0_EXTRACTION_V4.system.index("## surface"):])

    def test_kinship_and_name_variant_definitions_carried_over_from_v3(self):
        v3, v4 = MEM0_EXTRACTION_V3.system, MEM0_EXTRACTION_V4.system
        for key in ("kinship", "name_variant"):
            with self.subTest(key=key):
                start3, start4 = v3.index(f'- "{key}":'), v4.index(f'- "{key}":')
                self.assertEqual(v3[start3:start3 + 300], v4[start4:start4 + 300])

    def test_in_scope_examples_taken_unchanged_from_v3(self):
        v3 = dict(MEM0_EXTRACTION_V3.examples)
        for utterance, output in MEM0_EXTRACTION_V4.examples:
            with self.subTest(utterance=utterance):
                distinction = json.loads(output)["distinction"]
                if set(distinction) <= {"kinship", "register"}:
                    self.assertEqual(v3[utterance], output)  # byte-identical to v3
                else:  # name_variant-only: the Turkish examples drop their evidentiality key
                    self.assertEqual(set(distinction), {"name_variant"})
                    self.assertIn(utterance, v3)
                    old = json.loads(v3[utterance])["distinction"]
                    self.assertEqual(distinction["name_variant"], old["name_variant"])


if __name__ == "__main__":
    unittest.main()
