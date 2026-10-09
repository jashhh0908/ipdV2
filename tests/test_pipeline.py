"""Tests for the stage-attribution harness (dkmem.pipeline, Configs A-D) and its
parts: the merge judge, the Config B prompt/parser, the embedding metric and the
loss-point diagnosis. No GPU or model: a scripted fake generator answers the
extraction, native-extraction and judge prompts, and Config D uses a fake
similarity metric.

Run: python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from dkmem.backends.embedding import cosine, cosine_metric
from dkmem.backends.llm import GenerationParams
from dkmem.memory.judge import MERGE_JUDGE_V1, judge_pairs, parse_judge_output, render_judge_prompt
from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.native_extract import (
    MEM0_DEFAULT_FACTS_V1,
    NATIVE_B_V1,
    NATIVE_B_V1_SYSTEM_SHA256,
    NativeExtractionError,
    extraction_checks,
    native_extract_many,
    native_fact_extract_many,
    output_language_report,
    parse_fact_output,
    parse_facts_output,
)
from dkmem.memory.prompts import MEM0_EXTRACTION_V4
from dkmem.memory.similarity import SimilarityError, SimilarityMetric
from dkmem.pipeline.runner import (
    build_run_config,
    mode_slug,
    run_stage_attribution,
    strategy_for,
    write_outputs,
)
from dkmem.pipeline.trace import (
    config_fingerprint,
    diagnose,
    diagnose_dropped,
    inputs_hash,
    normalize_text,
    pair_input_hash,
    source_hashes,
)
from dkmem.pipeline import cli as pipeline_cli
from dkmem.tier1.io import Tier1Record

REPO = Path(__file__).parent.parent
LEXICON_PATH = REPO / "dkmem" / "memory" / "distinction_features.json"
PAIRWISE_SCHEMA = json.loads((REPO / "dkmem" / "pairwise_eval.schema.json").read_text(encoding="utf-8"))
MANIFEST_SCHEMA = json.loads((REPO / "dkmem" / "run_manifest.schema.json").read_text(encoding="utf-8"))

ALL_MODES = ["off", "lexicon", "lexicon+llm"]


def rec(pair="ep_1", a="meri chachi Pune mein rehti hai", b="meri mausi Pune mein rehti hai", lang="hi"):
    return Tier1Record(
        eval_pair_id=pair, language=lang, utterance_a=a, utterance_b=b,
        utterance_a_id=f"{pair}_utt_a", utterance_b_id=f"{pair}_utt_b",
        opaque_entity_id_a="ent_a", opaque_entity_id_b="ent_b",
    )


class FakeLLM:
    """Answers the three prompt kinds, selected by their system message.

    ``fact(u)`` is Config B's extractor: a string is wrapped as ``{"fact": ...}``;
    ``("raw", text)`` is returned verbatim (for malformed outputs).
    """

    backbone = "fake/qwen"

    def __init__(self, gloss=None, model_distinction=None, fact=None, judge=None):
        self.gloss = gloss or (lambda u: "user's aunt lives in Pune")
        self.model_distinction = model_distinction or (lambda u: {})
        self.fact = fact or (lambda u: u)
        self.judge = judge or (lambda a, b: ("merge", "same"))
        self.calls = {"v4": 0, "native": 0, "judge": 0}
        self.judge_prompts = []

    def generate(self, prompts, params=None):
        out = []
        for p in prompts:
            system, last = p[0]["content"], p[-1]["content"]
            if 'exactly one key and nothing else' in system:  # NATIVE_B_V1 (shares its opening with V4)
                self.calls["native"] += 1
                f = self.fact(last.removeprefix("Utterance: "))
                out.append(f[1] if isinstance(f, tuple) else json.dumps({"fact": f}, ensure_ascii=False))
            elif system.startswith("You convert one user utterance"):
                self.calls["v4"] += 1
                u = last.removeprefix("Utterance: ")
                out.append(json.dumps(
                    {"gloss": self.gloss(u), "distinction": self.model_distinction(u), "surface": u},
                    ensure_ascii=False))
            elif system.startswith("You maintain the long-term memory"):
                self.calls["judge"] += 1
                self.judge_prompts.append(p)
                a, b = last.split("\n")
                a, b = a.removeprefix("Memory A: "), b.removeprefix("Memory B: ")
                d = self.judge(a, b)
                out.append(d if isinstance(d, str) else json.dumps({"decision": d[0], "reason": d[1]}))
            else:
                raise AssertionError(f"unexpected prompt: {system[:40]!r}")
        return out


def run(config_id, records, llm, *, modes=ALL_MODES, similarity=None, tau=None, seed=0):
    lexicon = load_lexicon(LEXICON_PATH)
    params = GenerationParams(seed=seed)
    cfg = build_run_config(
        config_id, modes, records, generator=llm, lexicon_path=LEXICON_PATH, params=params,
        similarity=similarity, tau=tau,
    )
    return run_stage_attribution(
        records, config_id, lexicon, cfg, generator=llm, params=params, similarity=similarity, tau=tau
    )


def final(result, mode, i=0):
    return result.trace[i]["final"][mode]["pairwise_decision"]


class TestConfigA(unittest.TestCase):
    def test_english_gloss_collapse_is_lost_in_storage_and_gate_vetoes(self):
        llm = FakeLLM(judge=lambda a, b: ("merge", "identical") if a == b else ("keep_both", "differ"))
        res = run("A", [rec()], llm)
        t = res.trace[0]
        self.assertEqual(t["status"], "ok")
        self.assertEqual(t["stored"]["a"]["text"], t["stored"]["b"]["text"])  # chachi == mausi == "aunt"
        self.assertEqual(t["stored"]["a"]["storage"], "english_gloss")
        self.assertEqual(t["similarity"]["score"], 1.0)
        self.assertFalse(t["similarity"]["decisive"])
        self.assertEqual(t["host"]["proposed"], "merge")
        self.assertEqual(t["diagnosis"]["first_loss_point"], "storage")
        self.assertEqual(t["diagnosis"]["loss_stage"], "L1")
        self.assertEqual(final(res, "off"), "merge")
        self.assertEqual(final(res, "lexicon"), "no_merge")
        self.assertTrue(t["gate"]["lexicon"]["vetoed"])
        self.assertEqual(t["gate"]["lexicon"]["compatibility"], "incompatible")
        off_row = res.rows_by_mode["off"][0]
        self.assertIsNone(off_row.compatibility)
        self.assertIsNone(off_row.entry_a.distinction)
        self.assertEqual(off_row.strategy, "mem0")
        self.assertEqual(res.rows_by_mode["lexicon"][0].strategy, "dk-mem-lexicon")
        self.assertEqual(res.rows_by_mode["lexicon"][0].entry_a.distinction.value, "chachi")

    def test_judge_sees_only_gloss_and_is_shared_by_all_modes(self):
        llm = FakeLLM()
        res = run("A", [rec()], llm)
        self.assertEqual(llm.calls["judge"], 1)  # one judgement, three modes
        user_turn = llm.judge_prompts[0][-1]["content"]
        self.assertEqual(user_turn, "Memory A: user's aunt lives in Pune\nMemory B: user's aunt lives in Pune")
        self.assertNotIn("chachi", json.dumps(llm.judge_prompts[0]))
        self.assertEqual(set(res.rows_by_mode), set(ALL_MODES))

    def test_underdetermined_only_vetoes_a_proposed_merge(self):
        r = rec(b="meri aunty Pune mein rehti hai")  # kinship marked on one side only
        merged = run("A", [r], FakeLLM(judge=lambda a, b: ("merge", "x")))
        self.assertEqual(final(merged, "lexicon"), "underdetermined_link")
        self.assertEqual(merged.rows_by_mode["lexicon"][0].compatibility, "underdetermined")
        apart = run("A", [r], FakeLLM(judge=lambda a, b: ("keep_both", "x")))
        self.assertEqual(final(apart, "lexicon"), "no_merge")  # not turned into a link
        self.assertEqual(apart.rows_by_mode["lexicon"][0].compatibility, "underdetermined")

    def test_lexicon_llm_mode_does_not_use_the_model_where_the_lexicon_is_not_ambiguous(self):
        # Sec 5.3: the model is consulted only for spans the lexicon flags as ambiguous; a name the lexicon
        # does not cover is not tagged, and the model's kinship guess does not override the lexicon.
        r = rec(a="Mehmet dun eve geldi", b="meri chachi eve geldi", lang="hi")
        llm = FakeLLM(model_distinction=lambda u: {"name_variant": "Mehmet-latin", "kinship": "mausi"})
        res = run("A", [r], llm)
        g = res.trace[0]["gate"]
        self.assertEqual(g["lexicon"]["distinction_a"], {})
        self.assertEqual(g["lexicon+llm"]["distinction_a"], {})
        self.assertEqual(g["lexicon+llm"]["distinction_b"], {"kinship": "chachi"})
        self.assertEqual(g["lexicon+llm"], {**g["lexicon"], "applied": True})
        self.assertEqual(res.trace[0]["extraction"]["a"]["lexicon_ambiguous"], [])

    def test_lexicon_llm_mode_takes_the_models_value_for_a_lexicon_ambiguous_class(self):
        ambiguous = "meri chachi aur mausi Pune mein hain"  # two different kinship terms: the lexicon flags it
        r = rec(a=ambiguous, b="meri bua Pune mein hai")
        llm = FakeLLM(model_distinction=lambda u: {"kinship": "mausi", "register": "tu"} if u == ambiguous else {})
        res = run("A", [r], llm)
        t = res.trace[0]
        self.assertEqual(t["extraction"]["a"]["lexicon_ambiguous"], ["kinship"])
        self.assertEqual(t["gate"]["lexicon+llm"]["distinction_a"], {"kinship": "mausi"})  # the model's value; register ignored

    def test_the_model_is_called_for_the_gate_only_on_ambiguous_utterances(self):
        ambiguous = "meri chachi aur mausi Pune mein hain"
        records = [rec("ep_1", ambiguous, "meri bua Pune mein hai"), rec("ep_2", "mera bhai Pune mein hai", "mera bhai Delhi mein hai")]
        llm = FakeLLM(model_distinction=lambda u: {"kinship": "chachi"}, judge=lambda a, b: ("keep_both", "x"))
        res = run("C", records, llm)
        self.assertEqual(llm.calls["v4"], 1)  # one ambiguous utterance in four; Config C's host makes no V4 call
        self.assertEqual(res.trace[0]["extraction"]["a"]["model_distinction"], {"kinship": "chachi"})
        self.assertIsNone(res.trace[0]["extraction"]["b"]["model_distinction"])
        self.assertEqual(res.trace[0]["gate"]["lexicon+llm"]["distinction_a"], {"kinship": "chachi"})
        self.assertEqual(res.run_config["lexicon_llm_policy"], "model_only_for_lexicon_ambiguous_classes_v1")
        no_llm_mode = run("C", records, FakeLLM(), modes=["off", "lexicon"])
        self.assertNotIn("lexicon_llm_policy", no_llm_mode.run_config)

    def test_an_ambiguous_utterance_whose_v4_call_failed_is_not_gated_in_lexicon_llm_mode(self):
        ambiguous = "meri chachi aur mausi Pune mein hain"
        llm = FakeLLM()
        orig = llm.generate
        llm.generate = lambda prompts, params=None: ["not json" if (p[0]["content"].startswith("You convert") and ambiguous in p[-1]["content"]) else x
                                                   for p, x in zip(prompts, orig(prompts, params))]
        res = run("C", [rec("ep_1", ambiguous, "meri bua Pune mein hai")], llm)
        t = res.trace[0]
        self.assertEqual(t["gate"]["lexicon+llm"], {"applied": False, "skipped": "no V4 extraction for the model fallback"})
        self.assertIsNone(t["final"]["lexicon+llm"])
        self.assertEqual(len(res.rows_by_mode["lexicon+llm"]), 0)
        self.assertEqual(len(res.rows_by_mode["lexicon"]), 1)

    def test_extraction_failure_yields_trace_but_no_rows(self):
        llm = FakeLLM()
        orig = llm.generate
        llm.generate = lambda prompts, params=None: ["not json" if "Pune" in p[-1]["content"] else x
                                                   for p, x in zip(prompts, orig(prompts, params))]
        res = run("A", [rec()], llm)
        self.assertEqual(res.trace[0]["status"], "extraction_failed")
        self.assertEqual(res.trace[0]["extraction"]["a"]["raw_output"], "not json")
        self.assertEqual({m: len(r) for m, r in res.rows_by_mode.items()}, {m: 0 for m in ALL_MODES})
        self.assertEqual(res.summary["sides_extraction_failed"], 2)
        d = res.trace[0]["diagnosis"]
        self.assertEqual((d["first_loss_point"], d["loss_stage"], d["sides_dropped"]), ("dropped_at_extraction", "L1", ["a", "b"]))
        self.assertEqual(res.summary["first_loss_point"], {"dropped_at_extraction": 1})

    def test_judge_failure_is_not_turned_into_a_decision(self):
        res = run("A", [rec()], FakeLLM(judge=lambda a, b: "maybe?"))
        self.assertEqual(res.trace[0]["status"], "judge_failed")
        self.assertEqual(res.trace[0]["host"]["judge"]["raw_output"], "maybe?")
        self.assertIsNotNone(res.trace[0]["host"]["judge"]["error"])
        self.assertTrue(all(len(r) == 0 for r in res.rows_by_mode.values()))
        self.assertEqual(res.summary["pairs_judge_failed"], 1)

    def test_unsupported_language_is_skipped_everywhere(self):
        res = run("A", [rec(lang="x"), rec(pair="ep_2")], FakeLLM())
        self.assertEqual(res.trace[0]["status"], "unsupported_language")
        self.assertEqual(res.summary["episodes_skipped_unsupported_language"], 1)
        self.assertEqual(len(res.rows_by_mode["off"]), 1)


class TestConfigB(unittest.TestCase):
    DEVANAGARI = "\u092e\u0947\u0930\u0940 \u091a\u093e\u091a\u0940 \u092a\u0941\u0923\u0947 \u092e\u0947\u0902 \u0930\u0939\u0924\u0940 \u0939\u0948"

    def test_one_native_fact_per_side_is_stored_and_language_is_logged(self):
        def fact(u):
            return u if "chachi" in u else "User's aunt lives in Pune."

        llm = FakeLLM(fact=fact, judge=lambda a, b: ("merge", "x"))
        res = run("B", [rec()], llm)
        t = res.trace[0]
        self.assertEqual(t["stored"]["a"]["storage"], "extractor_output")
        self.assertEqual(t["stored"]["a"]["text"], "meri chachi Pune mein rehti hai")
        self.assertEqual(t["stored"]["a"]["output_language"]["label"], "source_language")
        self.assertEqual(t["stored"]["b"]["output_language"]["label"], "translated_to_english")
        self.assertEqual(t["stored"]["a"]["output_language"]["classifier_version"], "v2")
        self.assertTrue(t["stored"]["a"]["checks"]["exact_copy"])
        self.assertEqual(t["extraction"]["a"]["prompt_id"], NATIVE_B_V1.prompt_id)
        self.assertEqual(t["extraction"]["a"]["prompt_sha256"], NATIVE_B_V1.sha256)
        self.assertEqual(llm.calls["native"], 2)
        self.assertEqual(llm.calls["v4"], 0)  # no ambiguous utterance: lexicon+llm needs no model call; nothing stored
        # b's text no longer shows *mausi*, but it still differs from a's: not a storage loss
        d = t["diagnosis"]
        self.assertEqual((d["storage_state"], d["first_loss_point"], d["loss_stage"]), ("distinct_not_visible", "host_merge", "L3"))
        self.assertEqual(final(res, "lexicon"), "no_merge")
        self.assertEqual(len(res.trace), 1)  # exactly one candidate pair per episode

    def test_language_drift_script_drift_and_corruption_are_separate_outcomes(self):
        outputs = {
            "copy": lambda u: u,
            "english": lambda u: "User's aunt lives in Pune.",
            "script": lambda u: self.DEVANAGARI,
            "garbled": lambda u: "meri chachi Pu\ufffde mein rehti hai",
        }
        records = [rec(pair=f"ep_{k}", a=f"meri chachi Pune mein rehti hai", b=f"{k} meri mausi Pune mein rehti hai") for k in outputs]

        def fact(u):
            for k, fn in outputs.items():
                if u.startswith(k + " "):
                    return fn(u)
            return u  # side a: a verbatim copy

        res = run("B", records, FakeLLM(fact=fact), modes=["off"])
        labels = {t["eval_pair_id"]: t["stored"]["b"]["output_language"]["label"] for t in res.trace}
        self.assertEqual(labels["ep_english"], "translated_to_english")  # language drift
        self.assertEqual(labels["ep_script"], "script_changed")  # script drift
        sl, ck = res.summary["stored_language"], res.summary["extraction_checks"]
        self.assertEqual(sl["entries"], 8)
        self.assertEqual(sl["translated"], 1)
        self.assertEqual(sl["script_changed"], 1)
        self.assertEqual(sl["classifier_version"], "v2")
        self.assertEqual(ck["char_garble"], 1)  # corruption signal, independent of the language labels
        self.assertEqual(ck["any_corruption_signal"], 1)
        self.assertEqual(ck["exact_copy"], 5)  # the four side-a copies and the "copy" pair's b side
        self.assertIn("lower bound", ck["note"])

    def test_empty_or_malformed_fact_is_dropped_at_extraction(self):
        for bad in (("raw", '{"fact": ""}'), ("raw", "meri chachi Pune mein rehti hai"), ("raw", '{"facts": ["x"]}')):
            llm = FakeLLM(fact=lambda u, bad=bad: bad if "chachi" in u else u)
            res = run("B", [rec()], llm, modes=["off"])
            t = res.trace[0]
            self.assertEqual(t["status"], "extraction_failed")
            self.assertEqual(t["extraction"]["a"]["raw_output"], bad[1])
            self.assertTrue(t["extraction"]["a"]["dropped_at_extraction"])
            self.assertNotIn("dropped_at_extraction", t["extraction"]["b"])
            d = t["diagnosis"]
            self.assertEqual((d["first_loss_point"], d["loss_stage"], d["sides_dropped"]), ("dropped_at_extraction", "L2", ["a"]))
            self.assertTrue(d["lexicon_detectable_difference"])
            self.assertEqual(len(res.rows_by_mode["off"]), 0)
            self.assertEqual(res.summary["sides_extraction_failed"], 1)
            self.assertEqual(res.summary["first_loss_point"], {"dropped_at_extraction": 1})


class TestConfigC(unittest.TestCase):
    def test_verbatim_storage_judge_still_merges(self):
        llm = FakeLLM(judge=lambda a, b: ("merge", "both are about an aunt in Pune"))
        res = run("C", [rec()], llm, modes=["off", "lexicon"])
        t = res.trace[0]
        self.assertEqual(t["stored"]["a"]["text"], "meri chachi Pune mein rehti hai")
        self.assertEqual(t["stored"]["a"]["storage"], "surface_text")
        self.assertEqual(llm.calls["v4"], 0)  # no extraction at all
        self.assertEqual(t["diagnosis"]["first_loss_point"], "host_merge")
        self.assertEqual(t["diagnosis"]["loss_stage"], "L3")
        self.assertEqual(final(res, "off"), "merge")
        self.assertEqual(final(res, "lexicon"), "no_merge")
        self.assertEqual(res.rows_by_mode["off"][0].strategy, "store-surface-only")
        self.assertEqual(res.rows_by_mode["off"][0].entry_a.surface, "meri chachi Pune mein rehti hai")

    def test_judge_keeping_apart_is_preserved(self):
        res = run("C", [rec()], FakeLLM(judge=lambda a, b: ("keep_both", "different relatives")), modes=["off", "lexicon"])
        self.assertEqual(res.trace[0]["diagnosis"]["first_loss_point"], "none")
        self.assertEqual(final(res, "off"), "no_merge")
        self.assertEqual(final(res, "lexicon"), "no_merge")
        self.assertFalse(res.trace[0]["gate"]["lexicon"]["vetoed"])


class TestConfigD(unittest.TestCase):
    def metric(self, score=0.9):
        return SimilarityMetric("fake_embedding", lambda a, b: score)

    def test_threshold_decides_strictly_above_tau(self):
        for tau, expected in ((0.8, "merge"), (0.9, "no_merge"), (0.95, "no_merge")):
            res = run("D", [rec()], None, modes=["off"], similarity=self.metric(0.9), tau=tau)
            self.assertEqual(final(res, "off"), expected, tau)
            row = res.rows_by_mode["off"][0]
            self.assertEqual((row.threshold, row.similarity_score), (tau, 0.9))
            self.assertEqual(row.strategy, "embedding-threshold")
        t = res.trace[0]
        self.assertTrue(t["similarity"]["decisive"])
        self.assertTrue(t["host"]["tau_used"])
        self.assertIsNone(t["host"]["judge"])

    def test_gate_vetoes_an_embedding_merge(self):
        res = run("D", [rec()], None, modes=["off", "lexicon"], similarity=self.metric(0.97), tau=0.8)
        self.assertEqual(final(res, "off"), "merge")
        self.assertEqual(final(res, "lexicon"), "no_merge")
        self.assertEqual(res.trace[0]["diagnosis"]["loss_stage"], "L4")
        self.assertEqual(res.trace[0]["similarity"]["metric"], "fake_embedding")

    def test_needs_explicit_metric_and_tau(self):
        for kwargs in ({}, {"tau": 0.8}, {"similarity": self.metric()}):
            with self.assertRaises(ValueError):
                run("D", [rec()], None, modes=["off"], **kwargs)

    def test_lexicon_llm_needs_a_generator(self):
        with self.assertRaises(ValueError):
            run("D", [rec()], None, modes=["lexicon+llm"], similarity=self.metric(), tau=0.8)


class TestReproducibilityAndOutputs(unittest.TestCase):
    def test_same_inputs_same_fingerprint_and_trace(self):
        r = [rec()]
        a, b = run("A", r, FakeLLM()), run("A", r, FakeLLM())
        self.assertEqual(a.run_config["fingerprint"], b.run_config["fingerprint"])
        self.assertEqual(a.run_config["group_id"], b.run_config["group_id"])
        self.assertEqual(a.trace, b.trace)
        self.assertEqual([x.to_dict() for x in a.rows_by_mode["off"]], [x.to_dict() for x in b.rows_by_mode["off"]])

    def test_fingerprint_depends_on_seed_config_input_and_modes(self):
        base = run("A", [rec()], FakeLLM()).run_config
        self.assertNotEqual(base["fingerprint"], run("A", [rec()], FakeLLM(), seed=1).run_config["fingerprint"])
        self.assertNotEqual(base["fingerprint"], run("C", [rec()], FakeLLM()).run_config["fingerprint"])
        self.assertNotEqual(base["fingerprint"], run("A", [rec(a="meri bua Pune mein rehti hai")], FakeLLM()).run_config["fingerprint"])
        self.assertNotEqual(base["fingerprint"], run("A", [rec()], FakeLLM(), modes=["off"]).run_config["fingerprint"])

    def test_run_config_records_model_seed_config_and_input_hash(self):
        cfg = run("A", [rec()], FakeLLM(), seed=3).run_config
        self.assertEqual((cfg["seed"], cfg["backbone"]), (3, "fake/qwen"))
        self.assertEqual(cfg["pipeline_config"]["config_id"], "A")
        self.assertEqual(cfg["pipeline_config"]["isolates"], "L1")
        self.assertEqual(cfg["generation_params"]["seed"], 3)
        self.assertEqual(cfg["prompts"]["distinction_extraction"]["sha256"], MEM0_EXTRACTION_V4.sha256)
        self.assertEqual(cfg["prompts"]["merge_judge"]["sha256"], MERGE_JUDGE_V1.sha256)
        h = pair_input_hash("ep_1", "hi", rec().utterance_a, rec().utterance_b)
        self.assertEqual(cfg["input"]["inputs_sha256"], inputs_hash([h]))
        self.assertEqual(run("A", [rec()], FakeLLM()).trace[0]["input"]["input_sha256"], h)
        self.assertEqual(len(cfg["lexicon"]["sha256"]), 64)

    def test_input_hash_ignores_gold_ids(self):
        a = rec()
        b = Tier1Record(**{**a.__dict__, "opaque_entity_id_a": "other", "opaque_entity_id_b": "other2"})
        self.assertEqual(
            pair_input_hash(a.eval_pair_id, a.language, a.utterance_a, a.utterance_b),
            pair_input_hash(b.eval_pair_id, b.language, b.utterance_a, b.utterance_b),
        )

    def test_written_outputs_validate_against_team_b_schemas(self):
        for config_id, llm, kw in (
            ("A", FakeLLM(), {}),
            ("B", FakeLLM(), {}),
            ("C", FakeLLM(), {}),
            ("D", FakeLLM(), {"similarity": SimilarityMetric("fake", lambda a, b: 0.9), "tau": 0.8}),
        ):
            res = run(config_id, [rec(), rec(pair="ep_2", a="tu kya kar raha hai", b="aap kya kar rahe hain")], llm, **kw)
            with tempfile.TemporaryDirectory() as tmp:
                paths = write_outputs(Path(tmp) / "g", res)
                self.assertTrue(Path(paths["trace"]).exists())
                for mode in ALL_MODES:
                    d = Path(tmp) / "g" / mode_slug(mode)
                    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
                    jsonschema.validate(manifest, MANIFEST_SCHEMA)
                    self.assertEqual(manifest["pipeline_config"], config_id)
                    self.assertEqual(manifest["dkmem_mode"], mode)
                    self.assertEqual(manifest["seed"], 0)
                    lines = (d / "pairwise_eval.jsonl").read_text(encoding="utf-8").splitlines()
                    self.assertEqual(len(lines), 2)
                    for line in lines:
                        row = json.loads(line)
                        jsonschema.validate(row, PAIRWISE_SCHEMA)
                        self.assertEqual(row["run_id"], manifest["run_id"])
                        self.assertEqual(row["strategy"], manifest["strategy"])
                        self.assertEqual(row["compatibility"] is None, mode == "off")
                trace = [json.loads(l) for l in Path(paths["trace"]).read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(trace), 2)
                self.assertEqual(json.loads(Path(paths["run_config"]).read_text(encoding="utf-8"))["group_id"],
                                 res.run_config["group_id"])

    def test_trace_has_every_stage_from_input_to_final(self):
        t = run("A", [rec()], FakeLLM()).trace[0]
        for key in ("input", "extraction", "stored", "similarity", "host", "gate", "final", "diagnosis"):
            self.assertIn(key, t)
        self.assertEqual(t["input"]["utterance_a"], "meri chachi Pune mein rehti hai")
        self.assertIn("raw_output", t["extraction"]["a"])

    def test_strategy_mapping(self):
        self.assertEqual(
            [strategy_for(c, "off") for c in "ABCD"], ["mem0", "mem0", "store-surface-only", "embedding-threshold"]
        )
        self.assertEqual(strategy_for("C", "lexicon"), "dk-mem-lexicon")
        self.assertEqual(strategy_for("D", "lexicon+llm"), "dk-mem-lexicon-llm")
        with self.assertRaises(KeyError):
            strategy_for("E", "off")
        with self.assertRaises(ValueError):
            strategy_for("A", "on")


class TestHostThreshold(unittest.TestCase):
    def test_llm_judge_configs_have_no_threshold(self):
        for config_id in "ABC":
            res = run(config_id, [rec()], FakeLLM(), modes=["off", "lexicon"])
            cfg = res.run_config
            self.assertIsNone(cfg["tau"])
            self.assertFalse(cfg["tau_decides_merges"])
            self.assertIsNone(res.trace[0]["host"]["tau"])
            self.assertFalse(res.trace[0]["host"]["tau_used"])
            self.assertFalse(res.trace[0]["similarity"]["decisive"])
            for rows in res.rows_by_mode.values():
                self.assertIsNone(rows[0].threshold)
                d = rows[0].to_dict()
                self.assertIsNone(d["threshold"])
                jsonschema.validate(d, PAIRWISE_SCHEMA)
                self.assertIsInstance(d["similarity_score"], float)  # still logged, informational

    def test_tau_is_rejected_for_judge_configs(self):
        for config_id in "ABC":
            with self.assertRaises(ValueError):
                run(config_id, [rec()], FakeLLM(), tau=0.85)

    def test_config_d_rows_carry_their_tau(self):
        res = run("D", [rec()], None, modes=["off"], similarity=SimilarityMetric("m", lambda a, b: 0.9), tau=0.8)
        self.assertEqual(res.rows_by_mode["off"][0].threshold, 0.8)
        self.assertTrue(res.run_config["tau_decides_merges"])
        jsonschema.validate(res.rows_by_mode["off"][0].to_dict(), PAIRWISE_SCHEMA)

    def test_a_schema_without_null_would_reject_these_rows(self):
        schema = json.loads(json.dumps(PAIRWISE_SCHEMA))
        schema["properties"]["threshold"]["type"] = "number"
        row = run("C", [rec()], FakeLLM(), modes=["off"]).rows_by_mode["off"][0].to_dict()
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(row, schema)


class TestCodeHashes(unittest.TestCase):
    def test_run_config_records_source_hashes(self):
        cfg = run("C", [rec()], FakeLLM()).run_config
        code = cfg["code"]
        self.assertEqual(code["tree_sha256"], source_hashes()["tree_sha256"])
        files = code["files"]
        for rel in ("pipeline/runner.py", "memory/native_extract.py", "memory/distinction_features.json",
                    "pairwise_eval.schema.json", "run_manifest.schema.json", "backends/llm.py"):
            self.assertEqual(len(files[rel]), 64, rel)
        self.assertFalse([f for f in files if f.startswith("data/") or "__pycache__" in f])
        # outside the fingerprint on purpose (the run id must not change with a whitespace edit)
        outside = {"fingerprint", "group_id", "lexicon_path", "input_source_path", "created_at", "git", "code"}
        self.assertEqual(config_fingerprint({k: v for k, v in cfg.items() if k not in outside}), cfg["fingerprint"])

    def test_lexicon_hash_in_the_fingerprint_ignores_line_endings(self):
        import tempfile as tf
        text = LEXICON_PATH.read_bytes().replace(b"\r\n", b"\n")
        with tf.TemporaryDirectory() as tmp:
            lf, crlf = Path(tmp) / "lf.json", Path(tmp) / "crlf.json"
            lf.write_bytes(text)
            crlf.write_bytes(text.replace(b"\n", b"\r\n"))
            cfgs = []
            for path in (lf, crlf):
                cfgs.append(build_run_config("C", ["off"], [rec()], generator=FakeLLM(), lexicon_path=path,
                                             params=GenerationParams(seed=0)))
            self.assertEqual(cfgs[0]["lexicon"], cfgs[1]["lexicon"])
            self.assertEqual(cfgs[0]["fingerprint"], cfgs[1]["fingerprint"])

    def test_hashes_ignore_line_endings_and_track_content(self):
        import tempfile as tf
        with tf.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
            (root / "sub").mkdir()
            (root / "sub" / "b.json").write_bytes(b"{}")
            (root / "data").mkdir()
            (root / "data" / "skip.json").write_bytes(b"{}")
            first = source_hashes(root)
            self.assertEqual(sorted(first["files"]), ["a.py", "sub/b.json"])
            (root / "a.py").write_bytes(b"x = 1\ny = 2\n")
            self.assertEqual(source_hashes(root), first)
            (root / "a.py").write_bytes(b"x = 1\ny = 3\n")
            self.assertNotEqual(source_hashes(root)["tree_sha256"], first["tree_sha256"])


class TestCli(unittest.TestCase):
    def loaders(self):
        llm = FakeLLM()
        calls = {}

        def load_generator(model_id, revision, device, shard):
            calls["generator"] = (model_id, revision, device, shard)
            return llm

        class FakeEmbedder:
            def embed(self, texts):
                import math
                return [[math.cos(math.radians(sum(map(ord, t)) % 90)), math.sin(math.radians(sum(map(ord, t)) % 90))] for t in texts]

            def run_info(self):
                return {"model_config": {"model_id": "fake-embedder"}}

        def load_embedder(model_id, revision, device):
            calls["embedder"] = (model_id, revision, device)
            return FakeEmbedder()

        return llm, calls, load_generator, load_embedder

    def test_full_command_line_runs_a_to_d_and_writes_every_group(self):
        llm, calls, lg, le = self.loaders()
        with tempfile.TemporaryDirectory() as tmp:
            runs = pipeline_cli.main(
                ["--configs", "A", "B", "C", "D", "--modes", "off", "lexicon", "lexicon+llm",
                 "--model-id", "fake/qwen", "--seed", "3", "--tau-d", "0.5", "0.9999", "--limit", "4", "--out", tmp],
                load_generator=lg, load_embedder=le,
            )
            self.assertEqual([(r["config"], r["tau"]) for r in runs],
                             [("A", None), ("B", None), ("C", None), ("D", 0.5), ("D", 0.9999)])
            self.assertEqual(len({r["group_id"] for r in runs}), 5)
            self.assertEqual(calls["generator"], ("fake/qwen", None, "cuda:0", False))
            self.assertEqual(calls["embedder"][0], "BAAI/bge-m3")
            for r in runs:
                d = Path(r["dir"])
                for name in ("run_config.json", "trace.jsonl", "summary.json", "invocation.json"):
                    self.assertTrue((d / name).exists(), name)
                cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual(cfg["seed"], 3)
                self.assertEqual(cfg["input"]["limit"], 4)
                self.assertTrue(cfg["input"]["sanctioned"])
                self.assertEqual(cfg["input"]["file_sha256"], "eab5dcb23375eb2e744c72a98cb31c25da0171b2ba3297a0843574ed9a34cd07")
                self.assertEqual(cfg["tau"], r["tau"])
                self.assertEqual(len(cfg["code"]["tree_sha256"]), 64)
                for mode_dir in ("off", "lexicon", "lexicon-llm"):
                    man = json.loads((d / mode_dir / "run_manifest.json").read_text(encoding="utf-8"))
                    jsonschema.validate(man, MANIFEST_SCHEMA)
                    self.assertEqual(man["pipeline_config"], r["config"])
                    for line in (d / mode_dir / "pairwise_eval.jsonl").read_text(encoding="utf-8").splitlines():
                        row = json.loads(line)
                        jsonschema.validate(row, PAIRWISE_SCHEMA)
                        self.assertEqual(row["threshold"], r["tau"])
            d_runs = [r for r in runs if r["config"] == "D"]
            self.assertNotEqual(d_runs[0]["summary"]["final_decisions"], d_runs[1]["summary"]["final_decisions"])

    def test_shard_flag_and_validation(self):
        llm, calls, lg, le = self.loaders()
        with tempfile.TemporaryDirectory() as tmp:
            pipeline_cli.main(["--configs", "C", "--modes", "off", "--shard", "--limit", "1", "--out", tmp,
                               "--model-id", "Qwen/Qwen2.5-7B-Instruct"], load_generator=lg, load_embedder=le)
            self.assertEqual(calls["generator"], ("Qwen/Qwen2.5-7B-Instruct", None, "cuda:0", True))
            for argv in (["--configs", "D", "--out", tmp], ["--configs", "A", "--tau-d", "0.5", "--out", tmp],
                         ["--configs", "D", "--tau-d", "1.5", "--out", tmp]):
                with self.assertRaises(SystemExit):
                    pipeline_cli.main(argv, load_generator=lg, load_embedder=le)

    def test_config_d_without_llm_modes_loads_no_generator(self):
        llm, calls, lg, le = self.loaders()
        with tempfile.TemporaryDirectory() as tmp:
            pipeline_cli.main(["--configs", "D", "--modes", "off", "lexicon", "--tau-d", "0.8", "--limit", "2", "--out", tmp],
                              load_generator=lg, load_embedder=le)
        self.assertNotIn("generator", calls)


class TestJudge(unittest.TestCase):
    def test_parse(self):
        ok = parse_judge_output('{"decision": "keep_both", "reason": "x"}')
        self.assertEqual((ok.decision, ok.reason, ok.ok), ("keep_both", "x", True))
        for bad in ("merge", '{"decision": "merge"}', '{"decision": "supersede", "reason": "x"}',
                    '```json\n{"decision": "merge", "reason": "x"}\n```', '{"decision": "merge", "reason": 1}', "[]"):
            r = parse_judge_output(bad)
            self.assertFalse(r.ok, bad)
            self.assertIsNone(r.decision)
            self.assertEqual(r.raw_output, bad)

    def test_prompt_is_neutral_about_the_probe_classes(self):
        text = json.dumps(MERGE_JUDGE_V1.render("x"), ensure_ascii=False).lower()
        for word in ("kinship", "relative", "aunt", "uncle", "register", "honorific", "name", "language", "translat"):
            if word == "name":
                continue  # "name" never occurs in the system text; checked below
            self.assertNotIn(word, text, word)
        self.assertNotIn(" name", MERGE_JUDGE_V1.system.lower())

    def test_render_and_batch(self):
        msgs = render_judge_prompt("a text", "b text")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1], {"role": "user", "content": "Memory A: a text\nMemory B: b text"})
        out = judge_pairs([("x", "y"), ("p", "q")], FakeLLM(judge=lambda a, b: ("keep_both", a)))
        self.assertEqual([r.decision for r in out], ["keep_both", "keep_both"])
        self.assertEqual(judge_pairs([], FakeLLM()), [])


class TestNativeExtract(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_facts_output('{"facts": ["a", "b"]}'), (("a", "b"), False))
        self.assertEqual(parse_facts_output('{"facts": []}'), ((), False))
        self.assertEqual(parse_facts_output('```json\n{"facts": ["a"]}\n```'), (("a",), True))
        for bad in ("a fact", '{"fact": ["a"]}', '{"facts": "a"}', '{"facts": [""]}', '{"facts": [], "x": 1}', "[]"):
            with self.assertRaises(NativeExtractionError, msg=bad):
                parse_facts_output(bad)

    def test_prompt_matches_mem0_default_contract(self):
        self.assertEqual(MEM0_DEFAULT_FACTS_V1.output_keys, ("facts",))
        self.assertIn("record the facts in the same language", MEM0_DEFAULT_FACTS_V1.system)
        self.assertNotIn("English", MEM0_DEFAULT_FACTS_V1.system)  # no forced-English instruction
        self.assertNotEqual(MEM0_DEFAULT_FACTS_V1.sha256, MEM0_EXTRACTION_V4.sha256)
        self.assertEqual(MEM0_DEFAULT_FACTS_V1.render("namaste")[-1]["content"], "Input:\nuser: namaste\n")

    def test_native_b_v1_is_frozen(self):
        self.assertEqual(NATIVE_B_V1.prompt_id, "native_b_v1")
        self.assertEqual(NATIVE_B_V1.output_keys, ("fact",))
        self.assertEqual(NATIVE_B_V1.user_template, "Utterance: {utterance}")
        self.assertEqual(NATIVE_B_V1.examples, ())
        import hashlib
        self.assertEqual(hashlib.sha256(NATIVE_B_V1.system.encode("utf-8")).hexdigest(), NATIVE_B_V1_SYSTEM_SHA256)
        self.assertEqual(NATIVE_B_V1_SYSTEM_SHA256, "709d74689e30185022400bcc855adee3fb81fc8b305d1b2ea22ce8a4ab090601")
        self.assertEqual(NATIVE_B_V1.sha256, "4784de78239fa2a16eb18a4f604f8077ac5e40baf9c31f8d05c15a4f038525e2")

    def test_native_b_v1_is_the_validated_candidate_text(self):
        path = REPO / "validation" / "native_b_candidate" / "candidate_prompts.py"
        if not path.exists():
            self.skipTest("validation/native_b_candidate not present")
        ns = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        cand = ns["NATIVE_B_CANDIDATE"]
        self.assertEqual(cand.sha256, "6ca296ec04dc04c02bb7911618359f6ea82aaab25c92edec74d3bcfe635716af")
        self.assertEqual((NATIVE_B_V1.system, NATIVE_B_V1.user_template, NATIVE_B_V1.output_keys),
                         (cand.system, cand.user_template, cand.output_keys))

    def test_native_b_v1_is_not_the_mem0_baseline_and_instructs_native_output(self):
        text = NATIVE_B_V1.system
        self.assertIn("detect the language of the user input and record the fact in the same language", text)
        self.assertIn("Do not translate it into English", text)
        self.assertIn("Keep names and relationship terms exactly as written", text)
        self.assertIn("never return an empty string", text)
        self.assertNotEqual(NATIVE_B_V1.sha256, MEM0_DEFAULT_FACTS_V1.sha256)
        self.assertNotIn("Personal Information Organizer", text)
        self.assertNotIn("kinship", text.lower())  # no hint about the probe classes

    def test_parse_fact_output(self):
        self.assertEqual(parse_fact_output('{"fact": "Neha aa rahi hai"}'), "Neha aa rahi hai")
        for bad in ("Neha aa rahi hai", '{"fact": ""}', '{"fact": "  "}', '{"fact": 1}', '{"facts": ["x"]}',
                    '{"fact": "x", "y": 1}', '```json\n{"fact": "x"}\n```', "[]"):
            with self.assertRaises(NativeExtractionError, msg=bad):
                parse_fact_output(bad)

    def test_native_fact_extract_many(self):
        outs = iter(['{"fact": "a"}', "oops", '{"fact": ""}'])

        class G:
            backbone = "g"
            def generate(self, prompts, params=None):
                assert prompts[0][-1]["content"] == "Utterance: u1"
                return [next(outs) for _ in prompts]

        a, b, c = native_fact_extract_many(["u1", "u2", "u3"], G())
        self.assertEqual((a.ok, a.facts), (True, ("a",)))
        self.assertFalse(b.ok)
        self.assertEqual(b.raw_output, "oops")
        self.assertFalse(c.ok)
        self.assertEqual(native_fact_extract_many([], G()), [])

    def test_extraction_checks_are_signals_not_language_labels(self):
        u = "meri chachi Pune mein rehti hai"
        self.assertEqual(extraction_checks(u, u)["corruption_signals"], [])
        c = extraction_checks(u, "Meri chachi Pune mein rehti hai.")
        self.assertEqual((c["exact_copy"], c["normalized_copy"]), (False, True))
        self.assertTrue(extraction_checks(u, "meri chachi. Pune mein rehti hai")["multi_sentence"])
        self.assertTrue(extraction_checks(u, "Utterance: " + u)["label_leak"])
        for garbled in ("meri chachi Pu\ufffde mein", "\u0909\u0902CLE ne gaadi di", "\u0646ana ne cycle dilayi"):
            self.assertTrue(extraction_checks(u, garbled)["char_garble"], garbled)
        for fine in ("\u092e\u0947\u0930\u0940 \u091a\u093e\u091a\u0940 Pune mein", "Neha party mein aayi thi", "\ub531\uc774 \uacb0\ud63c\ud588\uc5b4\uc694"):
            self.assertFalse(extraction_checks(u, fine)["char_garble"], fine)
        # a Devanagari rewrite is script drift, not a corruption signal
        self.assertEqual(extraction_checks(u, "\u092e\u0947\u0930\u0940 \u091a\u093e\u091a\u0940")["corruption_signals"], [])

    def test_prompt_is_frozen(self):
        # Verified identical to upstream FACT_RETRIEVAL_PROMPT (date frozen) on 2026-10-07.
        self.assertEqual(
            MEM0_DEFAULT_FACTS_V1.sha256, "e9c4726fffc75febe5e5090b0d38305c76079165d54a9ee38e265c02877ad469"
        )

    def test_batch_keeps_failures_beside_successes(self):
        outs = iter(['{"facts": ["x"]}', "oops"])

        class G:
            backbone = "g"
            def generate(self, prompts, params=None):
                return [next(outs) for _ in prompts]

        a, b = native_extract_many(["u1", "u2"], G())
        self.assertTrue(a.ok)
        self.assertFalse(b.ok)
        self.assertEqual(b.raw_output, "oops")

    def test_v4_prompt_untouched(self):
        self.assertEqual(MEM0_EXTRACTION_V4.sha256, "3b4e1de73ac89d5142d0e0cfba52a18e03307f87ac69b2646fb1c9abbabdf00a")


class TestOutputLanguage(unittest.TestCase):
    def test_labels(self):
        rep = output_language_report("meri chachi Pune mein rehti hai", "Meri chachi Pune mein rehti hai", "hi")
        self.assertEqual(rep["label"], "source_language")
        self.assertEqual(rep["source_token_retention"], 1.0)
        rep = output_language_report("meri chachi Pune mein rehti hai", "User's aunt lives in Pune", "hi")
        self.assertEqual(rep["label"], "translated_to_english")
        rep = output_language_report("meri chachi Pune mein rehti hai", "User's chachi lives in Pune mein", "hi")
        self.assertEqual(rep["label"], "mixed")
        self.assertEqual(rep["classifier_version"], "v2")
        rep = output_language_report("मेरी चाची पुणे में रहती हैं", "मेरी चाची पुणे में रहती हैं", "hi")
        self.assertEqual((rep["label"], rep["source_script_share"]), ("source_language", 1.0))
        rep = output_language_report("मेरी चाची पुणे में रहती हैं", "User's aunt lives in Pune", "hi")
        self.assertEqual(rep["label"], "translated_to_english")
        self.assertEqual(output_language_report("my aunt lives in Pune", "User's aunt lives in Pune", "en")["label"],
                         "not_applicable")

    def test_a_surviving_name_does_not_count_as_a_kept_word(self):
        # v1 called these "mixed" (the name was a kept word); they are translations.
        cases = [
            ("Rohan ek botal doodh laaya", "Rohan gave one carton of milk."),
            ("Anjali aa rahi hai", "Anjali is coming."),
            ("Neha aa rahi hai", "Neha está llegando."),
            ("Markus, wie war dein Urlaub in Italien?", "Markus, how was your vacation in Italy?"),
            ("Ali geldi", "Ali came"),
        ]
        for src, out in cases:
            rep = output_language_report(src, out, "hi")
            self.assertEqual(rep["label"], "translated_to_english", out)
        # ... and a native fact that keeps the name and the words is still source-language
        for src in ("Rohan ek botal doodh laaya", "Markus, wie war dein Urlaub in Italien?", "Ali geldi"):
            self.assertEqual(output_language_report(src, src, "hi")["label"], "source_language", src)

    def test_loanwords_and_capitalized_words_are_not_evidence(self):
        rep = output_language_report("Verma uncle kal ghar aaye the", "Verma uncle came to the house", "hi")
        self.assertEqual(rep["content_words"], 3)  # kal ghar aaye ("the" is Hindi and English)
        self.assertEqual(rep["label"], "translated_to_english")
        self.assertIsNone(output_language_report("Neha party", "Neha party", "hi")["source_token_retention"])
        self.assertEqual(output_language_report("Neha party", "Neha party", "hi")["label"], "undetermined")

    def test_korean_and_devanagari_names_in_latin_text(self):
        self.assertEqual(output_language_report("저는 내일 부산에 갑니다", "저는 내일 부산에 갑니다", "ko")["label"], "source_language")
        self.assertEqual(output_language_report("저는 내일 부산에 갑니다", "User will go to Busan tomorrow", "ko")["label"],
                         "translated_to_english")
        # Hinglish rewritten in Devanagari keeps the language but not the script
        rep = output_language_report("Anjali aa rahi hai", "अंजलि आ रही है", "hi")
        self.assertEqual(rep["label"], "script_changed")
        # a Devanagari name is not a Latin content word; transliterating it is not drift
        rep = output_language_report("नेहा aa rahi hai", "Neha aa rahi hai", "hi")
        self.assertEqual(rep["label"], "source_language")


class TestDiagnose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = load_lexicon(LEXICON_PATH)

    def kw(self, **o):
        d = dict(lexicon=self.lex, lang="hi", utterance_a="meri chachi Pune mein rehti hai",
                 utterance_b="meri mausi Pune mein rehti hai",
                 stored_text_a="meri chachi Pune mein rehti hai", stored_text_b="meri mausi Pune mein rehti hai",
                 host_proposed="merge", storage_loss_stage="L1", host_loss_stage="L3")
        d.update(o)
        return d

    def test_loss_points(self):
        self.assertEqual(diagnose(**self.kw())["first_loss_point"], "host_merge")
        self.assertEqual(diagnose(**self.kw(host_proposed="keep_both"))["first_loss_point"], "none")
        d = diagnose(**self.kw(stored_text_a="user's aunt", stored_text_b="user's aunt"))
        self.assertEqual((d["first_loss_point"], d["loss_stage"]), ("storage", "L1"))
        self.assertFalse(d["distinction_survives_storage"])
        same = diagnose(**self.kw(utterance_b="meri chachi Pune mein rehti hai"))
        self.assertEqual(same["first_loss_point"], "no_detectable_difference")
        self.assertEqual(same["lexicon_compatibility"], "compatible")

    def test_degraded_but_distinct_storage_is_not_a_storage_loss(self):
        # mami -> "mother", mausi -> "aunt": the term is lost, the texts still differ
        kw = self.kw(utterance_a="meri mami Jaipur aa rahi hai", utterance_b="meri mausi Jaipur aa rahi hai",
                     stored_text_a="user's mother is coming to Jaipur", stored_text_b="user's aunt is coming to Jaipur")
        d = diagnose(**kw)
        self.assertEqual(d["lexicon_compatibility"], "incompatible")
        self.assertFalse(d["distinction_survives_storage"])
        self.assertFalse(d["stored_texts_identical"])
        self.assertEqual(d["storage_state"], "distinct_not_visible")
        self.assertEqual((d["first_loss_point"], d["loss_stage"]), ("host_merge", "L3"))
        kept_apart = diagnose(**{**kw, "host_proposed": "keep_both"})
        self.assertEqual((kept_apart["first_loss_point"], kept_apart["loss_stage"]), ("none", None))

    def test_collapsed_storage_is_the_only_storage_loss(self):
        for a, b in (("user's aunt", "user's aunt"), ("User's aunt.", "user's  aunt")):
            d = diagnose(**self.kw(stored_text_a=a, stored_text_b=b))
            self.assertEqual((d["storage_state"], d["first_loss_point"], d["loss_stage"]), ("collapsed", "storage", "L1"))
        d = diagnose(**self.kw(stored_text_a="meri chachi Pune mein rehti hai", stored_text_b="meri mausi Pune mein rehti hai"))
        self.assertEqual(d["storage_state"], "visible")
        self.assertEqual(normalize_text("User's  Aunt."), normalize_text("users aunt"))

    def test_dropped_at_extraction(self):
        d = diagnose_dropped(lexicon=self.lex, lang="hi", utterance_a="meri chachi Pune mein rehti hai",
                             utterance_b="meri mausi Pune mein rehti hai", extraction_loss_stage="L2", sides_dropped=["b", "a"])
        self.assertEqual((d["first_loss_point"], d["loss_stage"], d["sides_dropped"]), ("dropped_at_extraction", "L2", ["a", "b"]))
        self.assertTrue(d["lexicon_detectable_difference"])
        same = diagnose_dropped(lexicon=self.lex, lang="hi", utterance_a="meri chachi Pune mein rehti hai",
                                utterance_b="meri chachi Pune mein rehti hai", extraction_loss_stage="L1", sides_dropped=["a"])
        self.assertFalse(same["lexicon_detectable_difference"])

    def test_devanagari_stored_text_still_shows_the_distinction(self):
        d = diagnose(**self.kw(stored_text_a="मेरी चाची पुणे में रहती हैं", stored_text_b="मेरी मौसी पुणे में रहती हैं"))
        self.assertTrue(d["distinction_survives_storage"])
        self.assertEqual(d["first_loss_point"], "host_merge")


class TestEmbeddingMetric(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 0]), 0.0)
        for v in ([0.1] * 3, [0.1, 0.2, 0.3], [0.7, 0.3, 0.11, 0.9]):
            self.assertLessEqual(cosine(v, v), 1.0)  # rounding noise above 1 is not an error
            self.assertAlmostEqual(cosine(v, v), 1.0)
        with self.assertRaises(ValueError):
            cosine([1], [1, 2])

    def test_metric_embeds_each_text_once_and_is_symmetric(self):
        calls = []
        vecs = {"a": [1.0, 0.0], "b": [0.6, 0.8]}

        def embed(texts):
            calls.extend(texts)
            return [vecs[t] for t in texts]

        m = cosine_metric(embed, "fake_cos")
        self.assertAlmostEqual(m("a", "b"), 0.6)
        self.assertAlmostEqual(m("b", "a"), 0.6)
        self.assertEqual(sorted(calls), ["a", "b"])
        self.assertEqual(m.name, "fake_cos")

    def test_negative_cosine_is_an_error_not_clipped(self):
        m = cosine_metric(lambda ts: [[1.0, 0.0] if t == "a" else [-1.0, 0.0] for t in ts], "neg")
        with self.assertRaises(SimilarityError):
            m("a", "b")


if __name__ == "__main__":
    unittest.main()
