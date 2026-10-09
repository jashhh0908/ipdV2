"""Tests for the Mem0 OSS v1.0.11 reproduction (dkmem.baselines.mem0).

No GPU or network: a scripted fake LLM answers Mem0's extraction and update prompts and a hashing-bigram
embedder stands in for bge-m3. The upstream prompts and helper functions are the real vendored files (they are what
the reproduction executes), so these tests also pin them: hashes, the git blob ids GitHub reports for tag v1.0.11,
the frozen prompt date and the statements of the upstream add flow that the code mirrors.

Run: python -m unittest discover -s tests
"""

import ast
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import jsonschema

from dkmem.backends.llm import GenerationParams
from dkmem.baselines.mem0.events import MAPPING_ID, MAPPING_RULES, classify_pair, mapping_rules_sha256, side_state
from dkmem.baselines.mem0.memory import BatchEmbedder, GeneratorLLM, Mem0AddError, Mem0Error, Mem0Memory
from dkmem.baselines.mem0.runner import build_mem0_run_config, run_mem0_baseline, write_mem0_outputs
from dkmem.baselines.mem0.source import (
    FROZEN_PROMPT_DATE,
    MEM0_COMMIT,
    MEM0_TAG,
    UPSTREAM_DIR,
    UPSTREAM_PINS,
    Mem0SourceError,
    load_pinned,
    verify_structure,
    verify_upstream_pins,
)
from dkmem.tier1.io import Tier1Record

REPO = Path(__file__).parent.parent
PAIRWISE_SCHEMA = json.loads((REPO / "dkmem" / "pairwise_eval.schema.json").read_text(encoding="utf-8"))
MANIFEST_SCHEMA = json.loads((REPO / "dkmem" / "run_manifest.schema.json").read_text(encoding="utf-8"))

FACT_PROMPT_SHA256 = "3bbd191361a20a3d9bccc2d0aee551f7d5ae884433fc1ea88fd8ad87a5b5b84a"  # date 2026-10-07
UPDATE_PROMPT_SHA256 = "d68a6ef706f5477c156e69b40f55c81d9fdc17704952d679e9658da7e15749dd"


# --- fakes ---------------------------------------------------------------------------------------------------


def hashing_embed(texts):
    """Deterministic bigram-hashing vectors (L2-normalised): similar strings are close."""
    out = []
    for t in texts:
        v = [0.0] * 64
        s = t.casefold()
        for a, b in zip(s, s[1:]):
            v[(ord(a) * 31 + ord(b)) % 64] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out


def parse_update_prompt(prompt):
    """(old memories, new facts) from a rendered get_update_memory_messages prompt."""
    blocks = [ast.literal_eval(b.strip()) for b in re.findall(r"```\n(.*?)\n\s*```", prompt, re.S)]
    if "Current memory is empty." in prompt:
        return [], blocks[-1]
    return blocks[0], blocks[-1]


class FakeMem0LLM:
    """Scripted stand-in for Qwen: ``facts(utterance)`` -> raw extraction output, ``actions(old, facts)`` -> raw update output."""

    backbone = "fake/qwen"

    def __init__(self, facts=None, actions=None):
        self.facts = facts or (lambda u: json.dumps({"facts": [u]}))
        self.actions = actions or default_actions
        self.calls = []

    def generate(self, prompts, params=None):
        out = []
        for p in prompts:
            self.calls.append(p)
            if p[0]["role"] == "system" and p[0]["content"].startswith("You are a Personal Information Organizer"):
                out.append(self.facts(p[-1]["content"].removeprefix("Input:\nuser: ").rstrip("\n")))
            elif len(p) == 1 and "You are a smart memory manager" in p[0]["content"]:
                old, facts = parse_update_prompt(p[0]["content"])
                out.append(self.actions(old, facts))
            else:
                raise AssertionError(f"unexpected prompt: {p!r}"[:200])
        return out


def default_actions(old, facts):
    """Keyword-driven update LLM. Facts containing SUPERSEDE/DELETE/SAME act on memory "0"; HOSTFAIL breaks the output."""
    items = []
    used = False
    for f in facts:
        if "HOSTFAIL" in f:
            return "this is not json"
        if old and "SUPERSEDE" in f:
            items.append({"id": "0", "text": f, "event": "UPDATE", "old_memory": old[0]["text"]})
            used = True
        elif old and "DELETE" in f:
            items.append({"id": "0", "text": old[0]["text"], "event": "DELETE"})
            used = True
        elif old and "SAME" in f:
            items.append({"id": "0", "text": old[0]["text"], "event": "NONE"})
            used = True
        else:
            items.append({"id": str(len(old) + len(items)), "text": f, "event": "ADD"})
    if old and not used:
        for o in old:  # Mem0's own prompt lists unrelated memories as NONE
            items.append({"id": o["id"], "text": o["text"], "event": "NONE"})
    return json.dumps({"memory": items})


def make_memory(llm=None, **kw):
    llm = llm or FakeMem0LLM()
    adapter = GeneratorLLM(llm, GenerationParams(seed=0))
    return Mem0Memory(adapter, BatchEmbedder(hashing_embed), namespace="t", **kw), adapter


# --- the pinned upstream sources -------------------------------------------------------------------------------


class TestPinnedSources(unittest.TestCase):
    def test_pins_verify_and_name_the_release(self):
        pins = verify_upstream_pins()
        self.assertEqual(set(pins), set(UPSTREAM_PINS))
        self.assertEqual(MEM0_TAG, "v1.0.11")
        self.assertEqual(MEM0_COMMIT, "144627c4ce5bc4db6acac17cbd158065f2b27a8d")
        # the git blob ids GitHub reports for these three files at tag v1.0.11 (looked up when the pins were made)
        self.assertEqual(pins["mem0/memory/main.py"]["git_blob_sha1"], "1804e7831dddd1c0365dddac5dbef0c3b49f7269")
        self.assertEqual(pins["mem0/configs/prompts.py"]["git_blob_sha1"], "b7851a5105c978f301022b796511f42de10d27ef")
        self.assertEqual(pins["mem0/memory/utils.py"]["git_blob_sha1"], "61e3863e380972118d297b541e6fcb1e975f6abb")

    def test_a_changed_vendored_file_is_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(UPSTREAM_DIR, Path(d) / "u")
            target = Path(d) / "u" / "prompts.py.txt"
            target.write_bytes(target.read_bytes() + b"# tampered\n")
            with self.assertRaisesRegex(Mem0SourceError, "does not match"):
                verify_upstream_pins(Path(d) / "u")

    def test_a_missing_vendored_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(Mem0SourceError, "cannot read"):
                verify_upstream_pins(d)

    def test_crlf_checkout_of_the_vendored_files_still_matches(self):
        with tempfile.TemporaryDirectory() as d:
            for f in UPSTREAM_DIR.iterdir():
                (Path(d) / f.name).write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(set(verify_upstream_pins(d)), set(UPSTREAM_PINS))

    def test_add_path_structure_the_reproduction_mirrors(self):
        verify_structure()  # every mirrored statement is in the pinned main.py; no threshold in the add path

    def test_prompts_are_the_vendored_ones_with_a_frozen_date(self):
        p = load_pinned()
        self.assertEqual(p.provenance["frozen_prompt_date"], FROZEN_PROMPT_DATE)
        self.assertEqual(p.provenance["prompts"]["fact_extraction"]["sha256"], FACT_PROMPT_SHA256)
        self.assertEqual(p.provenance["prompts"]["update_memory"]["sha256"], UPDATE_PROMPT_SHA256)
        self.assertIn(f"Today's date is {FROZEN_PROMPT_DATE}.", p.fact_extraction_prompt)
        self.assertIn("record the facts in the same language", p.fact_extraction_prompt)
        other = load_pinned("2030-01-02")
        self.assertEqual(
            other.fact_extraction_prompt.replace("2030-01-02", FROZEN_PROMPT_DATE), p.fact_extraction_prompt
        )  # the date is the only thing that moves

    def test_helpers_behave_like_upstream(self):
        u = load_pinned().utils
        self.assertEqual(u.remove_code_blocks('```json\n{"facts": []}\n```'), '{"facts": []}')
        self.assertEqual(u.remove_code_blocks("<think>x</think>{}"), "{}")
        self.assertEqual(u.extract_json('sure! {"facts": ["a"]} bye'), '{"facts": ["a"]}')
        self.assertEqual(u.normalize_facts(["a", {"fact": "b"}, {"text": "c"}, {"x": 1}, ""]), ["a", "b", "c"])
        sys_prompt, user_prompt = u.get_fact_retrieval_messages("user: hi\n")
        self.assertEqual(user_prompt, "Input:\nuser: hi\n")
        self.assertEqual(u.ensure_json_instruction("s", "u"), ("s\n\nYou must return your response in valid JSON format with a 'facts' key containing an array of strings.", "u"))
        self.assertEqual(u.parse_messages([{"role": "user", "content": "x"}]), "user: x\n")


# --- the add flow ----------------------------------------------------------------------------------------------


class TestAddFlow(unittest.TestCase):
    def test_empty_memory_adds_the_extracted_fact_and_logs_the_raw_io(self):
        mem, adapter = make_memory()
        res = mem.add("meri chachi Pune mein rehti hai", user_id="u")
        t = res.trace
        self.assertEqual(t["extraction"]["facts"], ["meri chachi Pune mein rehti hai"])
        self.assertEqual(t["extraction"]["facts_path"], "direct")
        self.assertTrue(t["extraction"]["system_prompt"].startswith("You are a Personal Information Organizer"))
        self.assertEqual(t["extraction"]["user_prompt"], "Input:\nuser: meri chachi Pune mein rehti hai\n")
        self.assertEqual(t["extraction"]["response_format"], {"type": "json_object"})
        self.assertIn("Current memory is empty.", t["update"]["prompt"])
        self.assertEqual([a["outcome"] for a in t["actions"]], ["created"])
        self.assertEqual(len(res.results), 1)
        self.assertEqual(res.results[0]["event"], "ADD")
        self.assertEqual([h["event"] for h in res.history_rows], ["ADD"])
        self.assertEqual(res.history_rows[0]["old_memory"], None)
        self.assertEqual(mem.get_all("u")[0]["memory"], "meri chachi Pune mein rehti hai")
        self.assertEqual(len(adapter.calls), 2)  # extraction + update, each with raw messages and raw output
        self.assertIn("raw_output", adapter.calls[0])
        self.assertEqual(adapter.calls[1]["messages"][0]["role"], "user")  # the update call has no system message

    def test_existing_memories_are_remapped_to_small_integers_and_update_overwrites(self):
        mem, _ = make_memory()
        mem.add("lives in Pune", user_id="u")
        before = mem.get_all("u")[0]["id"]
        res = mem.add("SUPERSEDE lives in Delhi", user_id="u")
        t = res.trace
        self.assertEqual(t["retrieved_old_memory"], [{"id": "0", "text": "lives in Pune"}])  # ids hidden from the LLM
        self.assertEqual(t["id_mapping"], {"0": before})
        self.assertIn("'id': '0'", t["update"]["prompt"])
        self.assertNotIn(before, t["update"]["prompt"])
        self.assertEqual(t["actions"][0]["outcome"], "updated")
        self.assertEqual(t["actions"][0]["mapped_id"], before)
        self.assertEqual(mem.get_all("u"), [{"id": before, "memory": "SUPERSEDE lives in Delhi"}])
        self.assertEqual(res.results[0]["previous_memory"], "lives in Pune")
        hist = mem.history(before)
        self.assertEqual([h["event"] for h in hist], ["ADD", "UPDATE"])
        self.assertEqual((hist[1]["old_memory"], hist[1]["new_memory"]), ("lives in Pune", "SUPERSEDE lives in Delhi"))

    def test_delete_removes_the_memory_and_writes_a_history_row(self):
        mem, _ = make_memory()
        mem.add("likes tea", user_id="u")
        mid = mem.get_all("u")[0]["id"]
        res = mem.add("DELETE likes tea", user_id="u")
        self.assertEqual(res.trace["actions"][0]["outcome"], "deleted")
        self.assertEqual(mem.get_all("u"), [])
        last = mem.history(mid)[-1]
        self.assertEqual((last["event"], last["old_memory"], last["new_memory"], last["is_deleted"]), ("DELETE", "likes tea", None, 1))

    def test_none_changes_nothing(self):
        mem, _ = make_memory()
        mem.add("likes tea", user_id="u")
        res = mem.add("SAME likes tea", user_id="u")
        self.assertEqual([a["outcome"] for a in res.trace["actions"]], ["noop"])
        self.assertEqual(res.results, [])  # upstream does not report NONE
        self.assertEqual(len(mem.get_all("u")), 1)

    def test_actions_apply_sequentially_and_a_bad_one_does_not_stop_the_rest(self):
        def actions(old, facts):
            return json.dumps({"memory": [
                {"id": "9", "text": "x", "event": "UPDATE"},          # hallucinated id -> KeyError, skipped
                {"text": "", "event": "ADD"},                            # empty text -> skipped
                {"id": "5", "text": "new fact", "event": "ADD"},       # applied
                {"id": "5", "text": "weird", "event": "FROBNICATE"},   # unknown event -> nothing
                "not a dict",                                           # AttributeError inside the guard
                {"id": "0", "text": "later", "event": "ADD"},          # still applied
            ]})
        mem, _ = make_memory(FakeMem0LLM(actions=actions))
        res = mem.add("anything", user_id="u")
        self.assertEqual([a["outcome"] for a in res.trace["actions"]],
                         ["error", "skipped_empty_text", "created", "ignored_unknown_event", "error", "created"])
        self.assertIn("KeyError", res.trace["actions"][0]["error"])
        self.assertEqual({m["memory"] for m in mem.get_all("u")}, {"new fact", "later"})

    def test_unparseable_extraction_becomes_no_facts_and_no_update_call(self):
        mem, adapter = make_memory(FakeMem0LLM(facts=lambda u: "I cannot do that"))
        res = mem.add("x", user_id="u")
        self.assertEqual(res.trace["extraction"]["facts"], [])
        self.assertIn("JSONDecodeError", res.trace["extraction"]["facts_error"])
        self.assertFalse(res.trace["update"]["called"])
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(side_state(res.trace), "extraction_parse_failed")

    def test_missing_facts_key_is_also_swallowed(self):
        mem, _ = make_memory(FakeMem0LLM(facts=lambda u: json.dumps({"memories": ["a"]})))
        res = mem.add("x", user_id="u")
        self.assertEqual(res.trace["extraction"]["facts"], [])
        self.assertIn("KeyError", res.trace["extraction"]["facts_error"])

    def test_empty_facts_list_skips_the_update_call(self):
        mem, adapter = make_memory(FakeMem0LLM(facts=lambda u: '{"facts": []}'))
        res = mem.add("hi", user_id="u")
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(side_state(res.trace), "no_facts")

    def test_code_fences_and_chatty_output_are_handled_like_upstream(self):
        mem, _ = make_memory(FakeMem0LLM(facts=lambda u: '```json\n{"facts": ["a fact"]}\n```'))
        self.assertEqual(mem.add("x", user_id="u").trace["extraction"]["facts"], ["a fact"])
        mem, _ = make_memory(FakeMem0LLM(facts=lambda u: 'Sure! {"facts": ["b fact"]} hope that helps'))
        t = mem.add("x", user_id="u").trace
        self.assertEqual((t["extraction"]["facts"], t["extraction"]["facts_path"]), (["b fact"], "extract_json"))

    def test_object_shaped_facts_are_normalised(self):
        mem, _ = make_memory(FakeMem0LLM(facts=lambda u: json.dumps({"facts": [{"fact": "obj fact"}, "str fact"]})))
        self.assertEqual(mem.add("x", user_id="u").trace["extraction"]["facts"], ["obj fact", "str fact"])

    def test_update_llm_failure_stores_nothing_and_is_reported(self):
        class Breaks(FakeMem0LLM):
            def generate(self, prompts, params=None):
                if len(prompts[0]) == 1:
                    raise RuntimeError("cuda out of memory")
                return super().generate(prompts, params)

        mem, _ = make_memory(Breaks())
        res = mem.add("x", user_id="u")
        self.assertIn("cuda out of memory", res.trace["update"]["llm_error"])
        self.assertEqual(mem.get_all("u"), [])
        self.assertEqual(side_state(res.trace), "update_failed")

    def test_update_with_invalid_json_stores_nothing(self):
        mem, _ = make_memory(FakeMem0LLM(actions=lambda o, f: "ADD everything"))
        res = mem.add("x", user_id="u")
        self.assertIsNotNone(res.trace["update"]["parse_error"])
        self.assertEqual(res.trace["actions"], [])
        self.assertEqual(mem.get_all("u"), [])

    def test_extraction_llm_exception_is_not_swallowed(self):
        class Breaks(FakeMem0LLM):
            def generate(self, prompts, params=None):
                raise RuntimeError("boom")

        mem, _ = make_memory(Breaks())
        with self.assertRaises(Mem0AddError) as cm:
            mem.add("x", user_id="u")
        self.assertIn("boom", cm.exception.trace["exception"])
        self.assertEqual(side_state(cm.exception.trace), "llm_exception")
        self.assertIn("system_prompt", cm.exception.trace["extraction"])

    def test_search_is_limit_5_with_no_score_threshold(self):
        mem, _ = make_memory()
        for i in range(7):
            mem.add(f"completely different topic number {i} zzzz{i * 7}", user_id="u")
        res = mem.add("SAME qqq", user_id="u")
        hits = res.trace["retrieval"][0]["hits"]
        self.assertEqual(len(hits), 5)  # 7 memories exist; limit=5
        self.assertEqual(len(res.trace["retrieved_old_memory"]), 5)
        self.assertEqual([h["score"] for h in hits], sorted((h["score"] for h in hits), reverse=True))
        self.assertLess(min(h["score"] for h in hits), 0.9)  # weakly related memories are still passed to the LLM
        with self.assertRaises(Mem0Error):
            Mem0Memory(GeneratorLLM(FakeMem0LLM()), BatchEmbedder(hashing_embed), search_limit=3)

    def test_memories_hit_by_several_facts_are_listed_once(self):
        mem, _ = make_memory(FakeMem0LLM(facts=lambda u: json.dumps({"facts": ["lives in Pune", "lives in Pune city"]})))
        mem.add("lives in Pune", user_id="u")
        res = mem.add("again", user_id="u")
        self.assertEqual(len(res.trace["retrieved_old_memory"]), len(set(m["id"] for m in res.trace["retrieved_old_memory"])))
        self.assertEqual(len(res.trace["retrieval"]), 2)

    def test_memories_are_scoped_to_the_user(self):
        mem, _ = make_memory()
        mem.add("alpha fact", user_id="u1")
        res = mem.add("alpha fact again", user_id="u2")
        self.assertEqual(res.trace["retrieved_old_memory"], [])

    def test_unsupported_inputs_are_refused(self):
        mem, _ = make_memory()
        with self.assertRaises(Mem0Error):
            mem.add("x", user_id="u", infer=False)
        with self.assertRaises(Mem0Error):
            mem.add("x", user_id="")
        with self.assertRaises(Mem0Error):
            mem.add([{"role": "user", "content": ["image"]}], user_id="u")

    def test_runs_are_deterministic(self):
        def run():
            mem, _ = make_memory()
            mem.add("one fact", user_id="u")
            return mem.add("SUPERSEDE two fact", user_id="u").trace, mem.get_all("u"), mem.db.rows

        self.assertEqual(run(), run())


# --- event mapping ---------------------------------------------------------------------------------------------


def trace(facts, actions=None, *, update_error=None, parsed=True, extraction_error=None):
    acts = [dict({"index": i, "event": None, "id": None, "mapped_id": None, "text": None, "outcome": None,
                  "memory_id": None, "error": None}, **a) for i, a in enumerate(actions or [])]
    return {
        "extraction": {"facts": facts, "facts_error": extraction_error, "llm_error": None},
        "update": {"called": bool(facts), "llm_error": update_error, "parse_error": None,
                   "parsed": {"memory": []} if parsed else {}},
        "actions": acts,
    }


MAPPING_RULES_SHA256 = "2e10b333f63076bf47123e85acafdafc8514c48c3a16922b3ba92f758986992c"


class TestPairMapping(unittest.TestCase):
    def test_the_mapping_is_frozen(self):
        # Changing any rule changes this hash: it needs a new MAPPING_ID and a deliberate re-pin, not a silent edit.
        self.assertEqual(MAPPING_ID, "mem0_pair_mapping_v1")
        self.assertEqual(mapping_rules_sha256(), MAPPING_RULES_SHA256)
        self.assertEqual(len(MAPPING_RULES), 9)

    def test_run_config_carries_the_mapping_hash(self):
        cfg, _, _ = run_records(RECORDS[:2])
        self.assertEqual(cfg["baseline"]["mapping_sha256"], MAPPING_RULES_SHA256)

    A = trace(["a"], [{"event": "ADD", "outcome": "created", "memory_id": "m1", "text": "a"}])

    def test_update_or_delete_of_a_memory_is_supersede(self):
        for event, outcome in (("UPDATE", "updated"), ("DELETE", "deleted")):
            b = trace(["bf"], [{"event": event, "outcome": outcome, "memory_id": "m1", "mapped_id": "m1", "text": "t"}])
            p = classify_pair(self.A, b, a_memory_ids=["m1"])
            self.assertEqual((p.outcome, p.a_memory_id, p.entry_b_text), ("supersede", "m1", "bf"))

    def test_add_is_keep_both_even_when_the_llm_lists_a_as_none(self):
        b = trace(["bf"], [
            {"event": "NONE", "outcome": "noop", "mapped_id": "m1"},
            {"event": "ADD", "outcome": "created", "memory_id": "m2", "text": "bf2"},
        ])
        p = classify_pair(self.A, b, a_memory_ids=["m1"])
        self.assertEqual((p.outcome, p.entry_b_text, p.b_memory_id), ("keep_both", "bf2", "m2"))

    def test_none_alone_is_merge(self):
        b = trace(["bf"], [{"event": "NONE", "outcome": "noop", "mapped_id": "m1"}])
        self.assertEqual(classify_pair(self.A, b, a_memory_ids=["m1"]).outcome, "merge")

    def test_supersede_outranks_keep_both_and_merge(self):
        b = trace(["bf"], [
            {"event": "ADD", "outcome": "created", "memory_id": "m2", "text": "x"},
            {"event": "UPDATE", "outcome": "updated", "memory_id": "m1", "mapped_id": "m1"},
        ])
        self.assertEqual(classify_pair(self.A, b, a_memory_ids=["m1"]).outcome, "supersede")

    def test_failed_actions_do_not_count(self):
        b = trace(["bf"], [{"event": "UPDATE", "outcome": "error", "mapped_id": None, "error": "KeyError"}])
        p = classify_pair(self.A, b, a_memory_ids=["m1"])
        self.assertEqual(p.outcome, "host_failed")

    def test_no_facts_is_no_entry_and_outranks_a_failed_host(self):
        b_empty = trace([])
        self.assertEqual(classify_pair(self.A, b_empty, a_memory_ids=["m1"]).outcome, "no_entry")
        a_empty = trace([])
        self.assertEqual(classify_pair(a_empty, trace(["x"], parsed=False), a_memory_ids=[]).outcome, "no_entry")
        self.assertEqual(classify_pair(self.A, trace([], extraction_error="JSONDecodeError"), a_memory_ids=["m1"]).outcome, "no_entry")

    def test_unusable_update_step_is_host_failed(self):
        self.assertEqual(classify_pair(self.A, trace(["x"], parsed=False), a_memory_ids=["m1"]).outcome, "host_failed")
        self.assertEqual(classify_pair(self.A, trace(["x"], update_error="oom"), a_memory_ids=["m1"]).outcome, "host_failed")
        self.assertEqual(classify_pair(None, self.A, a_memory_ids=[]).outcome, "host_failed")  # a: LLM exception
        a_nothing = trace(["a"], [])  # facts but the update stored nothing
        self.assertEqual(classify_pair(a_nothing, self.A, a_memory_ids=[]).a_state, "nothing_stored")

    def test_keep_both_picks_the_best_matching_a_memory(self):
        a2 = trace(["a", "a2"], [
            {"event": "ADD", "outcome": "created", "memory_id": "m1", "text": "a"},
            {"event": "ADD", "outcome": "created", "memory_id": "m2", "text": "a2"},
        ])
        b = trace(["bf"], [{"event": "ADD", "outcome": "created", "memory_id": "m3", "text": "bf"}])
        self.assertEqual(classify_pair(a2, b, a_memory_ids=["m1", "m2"], retrieval_scores={"m1": 0.2, "m2": 0.7}).a_memory_id, "m2")
        self.assertEqual(classify_pair(a2, b, a_memory_ids=["m1", "m2"]).a_memory_id, "m1")


# --- the Tier 1 runner ------------------------------------------------------------------------------------------


def rec(pair, a, b, lang="hi"):
    return Tier1Record(
        eval_pair_id=pair, language=lang, utterance_a=a, utterance_b=b,
        utterance_a_id=f"{pair}_utt_a", utterance_b_id=f"{pair}_utt_b",
        opaque_entity_id_a=f"ent_{pair}_a", opaque_entity_id_b=f"ent_{pair}_b",
    )


RECORDS = [
    rec("ep_1", "meri chachi Pune mein rehti hai", "meri mausi Delhi mein rehti hai"),          # ADD -> keep_both
    rec("ep_2", "mera bhai Pune mein hai", "SUPERSEDE mera bhai Delhi mein hai"),              # UPDATE -> supersede
    rec("ep_3", "mera bhai Pune mein hai", "SAME mera bhai Pune mein hai"),                    # NONE -> merge
    rec("ep_4", "mera bhai Pune mein hai", "DELETE mera bhai Pune mein hai"),                  # DELETE -> supersede
    rec("ep_5", "NOFACTS hello", "mera bhai Pune mein hai"),                                    # a empty -> no_entry
    rec("ep_6", "mera bhai Pune mein hai", "HOSTFAIL mera bhai"),                               # update unparseable -> host_failed
    rec("ep_7", "meri chachi Pune mein hai", "meri mausi Pune mein hai", lang="x"),            # unsupported language
]


def facts_fn(u):
    return '{"facts": []}' if "NOFACTS" in u else json.dumps({"facts": [u]}, ensure_ascii=False)


def run_records(records=RECORDS, llm=None):
    llm = llm or FakeMem0LLM(facts=facts_fn)
    params = GenerationParams(seed=0)
    cfg = build_mem0_run_config(records, generator=llm, params=params, embedder_info={"model_config": {"model_id": "fake"}},
                                input_info={"file_sha256": "x", "sanctioned": False})
    return cfg, run_mem0_baseline(records, cfg, generator=llm, params=params, embed_texts=hashing_embed), llm


class TestMem0Runner(unittest.TestCase):
    def test_outcomes_rows_and_failure_policy(self):
        cfg, res, _ = run_records()
        by_id = {t["eval_pair_id"]: t for t in res.trace}
        self.assertEqual({k: v["status"] for k, v in by_id.items()},
                         {"ep_1": "ok", "ep_2": "ok", "ep_3": "ok", "ep_4": "ok", "ep_5": "no_entry",
                          "ep_6": "host_failed", "ep_7": "unsupported_language"})
        self.assertEqual({t["eval_pair_id"]: t["pair"]["outcome"] for t in res.trace if "pair" in t},
                         {"ep_1": "keep_both", "ep_2": "supersede", "ep_3": "merge", "ep_4": "supersede",
                          "ep_5": "no_entry", "ep_6": "host_failed"})
        rows = {r.record_id.split("_utt_a")[0]: r for r in res.rows}
        self.assertEqual(sorted(rows), ["ep_1", "ep_2", "ep_3", "ep_4"])  # failures and no_entry make no row
        self.assertEqual([r.decision for r in res.rows], ["no_merge", "supersede", "merge", "supersede"])
        r2 = rows["ep_2"]
        self.assertEqual(r2.superseded_entry_id, r2.entry_a.entry_id)
        self.assertEqual(r2.predicted_entity_id, r2.entry_a.entry_id)
        for r in res.rows:
            self.assertEqual((r.strategy, r.threshold, r.compatibility), ("mem0", None, None))
            self.assertEqual(r.entry_a.gold_entity_id, f"ent_{r.entry_a.source_utterance_id.split('_utt')[0]}_a")
            self.assertTrue(0 <= r.similarity_score <= 1)
        self.assertEqual(rows["ep_2"].entry_a.gloss, "mera bhai Pune mein hai")  # the text a held when b arrived
        self.assertEqual(res.summary["episodes_skipped_unsupported_language"], 1)
        self.assertEqual(res.summary["pair_outcomes"], {"host_failed": 1, "keep_both": 1, "merge": 1, "no_entry": 1, "supersede": 2})
        self.assertEqual(res.summary["final_decisions"], {"off": {"merge": 1, "no_merge": 1, "supersede": 2}})

    def test_trace_keeps_raw_llm_io_and_history(self):
        _, res, _ = run_records()
        t = next(t for t in res.trace if t["eval_pair_id"] == "ep_2")
        self.assertEqual(t["schema"], "dkmem_mem0_trace_v1")
        self.assertTrue(t["add_b"]["extraction"]["system_prompt"].startswith("You are a Personal Information Organizer"))
        self.assertIn("SUPERSEDE", t["add_b"]["extraction"]["raw_output"])
        self.assertIn("old_memory", t["add_b"]["update"]["raw_output"])
        self.assertEqual([h["event"] for h in t["add_b"]["history"]], ["UPDATE"])
        self.assertEqual([h["event"] for h in t["add_a"]["history"]], ["ADD"])
        self.assertEqual(t["memory_final"][0]["memory"], "SUPERSEDE mera bhai Delhi mein hai")
        self.assertEqual(t["pair"]["mapping"], "mem0_pair_mapping_v1")
        json.dumps(res.trace, allow_nan=False)  # fully serialisable

    def test_run_config_records_the_pinned_source_and_prompts(self):
        cfg, _, _ = run_records()
        m = cfg["mem0"]
        self.assertEqual((m["tag"], m["commit"], m["version"]), ("v1.0.11", "144627c4ce5bc4db6acac17cbd158065f2b27a8d", "1.0.11"))
        self.assertEqual(set(m["files"]), set(UPSTREAM_PINS))
        self.assertEqual((m["search_limit"], m["search_threshold"], m["frozen_prompt_date"]), (5, None, FROZEN_PROMPT_DATE))
        self.assertEqual(cfg["prompts"]["mem0_fact_extraction"]["sha256"], FACT_PROMPT_SHA256)
        self.assertEqual(cfg["prompts"]["mem0_update_memory"]["sha256"], UPDATE_PROMPT_SHA256)
        self.assertIn("substitutions", m)
        self.assertTrue(cfg["group_id"].startswith("mem0-v1.0.11-fake-qwen-s0-"))
        self.assertFalse(cfg["tau_decides_merges"])
        self.assertIsNone(cfg["tau"])

    def test_fingerprint_is_reproducible_and_input_sensitive(self):
        cfg1, res1, _ = run_records()
        cfg2, res2, _ = run_records()
        self.assertEqual(cfg1["fingerprint"], cfg2["fingerprint"])
        self.assertEqual(res1.trace, res2.trace)  # greedy fake: identical traces, ids and timestamps included
        self.assertEqual([r.to_dict() for r in res1.rows if True], [r.to_dict() for r in res2.rows])
        cfg3, _, _ = run_records(RECORDS[:3])
        self.assertNotEqual(cfg1["fingerprint"], cfg3["fingerprint"])

    def test_outputs_validate_against_the_team_b_schemas(self):
        _, res, _ = run_records()
        with tempfile.TemporaryDirectory() as d:
            paths = write_mem0_outputs(Path(d) / res.run_config["group_id"], res)
            rows = [json.loads(l) for l in Path(paths["off/pairwise_eval"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 4)
            for row in rows:
                jsonschema.validate(row, PAIRWISE_SCHEMA)
            manifest = json.loads((Path(paths["off/pairwise_eval"]).parent / "run_manifest.json").read_text(encoding="utf-8"))
            jsonschema.validate(manifest, MANIFEST_SCHEMA)
            self.assertEqual((manifest["strategy"], manifest["dkmem_mode"]), ("mem0", "off"))
            self.assertNotIn("pipeline_config", manifest)
            self.assertTrue(Path(paths["trace"]).exists() and Path(paths["summary"]).exists())

    def test_duplicate_episode_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            run_records([RECORDS[0], RECORDS[0]])

    def test_the_baseline_uses_the_pinned_prompts_verbatim(self):
        _, res, llm = run_records(RECORDS[:2])
        pinned = load_pinned()
        first = llm.calls[0]
        self.assertEqual(first[0]["content"], pinned.fact_extraction_prompt)  # json is already in it: no ensure_json suffix
        update = next(c for c in llm.calls if len(c) == 1)
        self.assertTrue(update[0]["content"].startswith(pinned.update_memory_prompt))


if __name__ == "__main__":
    unittest.main()
