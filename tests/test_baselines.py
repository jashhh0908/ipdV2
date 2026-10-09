"""Tests for the prompt-informed merge judge, the never-merge / flat-RAG baseline (Tier 1 and Tier 2 infrastructure),
the baselines CLI, and the cross-system fairness of the A-D, Mem0 and flat outputs.

Run: python -m unittest discover -s tests   (test_pipeline / test_mem0_baseline / test_eval helpers are imported by name)
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

import jsonschema
from test_eval import GOLD, RECORDS, fake_embedding_metric, run_d
from test_mem0_baseline import FakeMem0LLM, facts_fn, hashing_embed
from test_pipeline import LEXICON_PATH, FakeLLM, rec

from dkmem.backends.llm import GenerationParams
from dkmem.baselines import cli as baselines_cli
from dkmem.baselines.flat import (
    FLAT_QA_V1,
    FlatRAG,
    NeverMergeHost,
    Question,
    Turn,
    answer_questions,
    build_flat_run_config,
    ingest_conversation,
    normalize_answer,
    qa_report,
    retrieval_report,
    run_flat_baseline,
    token_f1,
    write_flat_outputs,
)
from dkmem.baselines.mem0.runner import build_mem0_run_config, run_mem0_baseline, write_mem0_outputs
from dkmem.eval.fairness import check_comparable
from dkmem.eval.groups import load_group
from dkmem.eval.metrics import score
from dkmem.memory.judge import JUDGE_PROMPTS, MERGE_JUDGE_INFORMED_V1, MERGE_JUDGE_V1
from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.similarity import SimilarityMetric
from dkmem.pipeline import cli as pipeline_cli
from dkmem.pipeline.runner import build_run_config, run_stage_attribution, strategy_for, write_outputs
from dkmem.store import cli as store_cli
from dkmem.store.consolidate import ConsolidationConfig, JudgeHost, ThresholdHost
from dkmem.store.runner import build_store_run_config, run_store_attribution

REPO = Path(__file__).parent.parent
PAIRWISE_SCHEMA = json.loads((REPO / "dkmem" / "pairwise_eval.schema.json").read_text(encoding="utf-8"))
MANIFEST_SCHEMA = json.loads((REPO / "dkmem" / "run_manifest.schema.json").read_text(encoding="utf-8"))
ALL_MODES = ["off", "lexicon", "lexicon+llm"]

INFORMED_SHA256 = "614f23318c07c411135cafdc61893961d01cf3c82c2e1d4422b0ce1b467bd1ad"
JUDGE_V1_SHA256 = "3855ef65f50e9a0f017710e5c0d0cb5a63d210a7547fa0e0e336d4c77a45c5ff"


# --- the prompt-informed judge ---------------------------------------------------------------------------------


class TestInformedJudgePrompt(unittest.TestCase):
    def test_both_prompts_are_frozen(self):
        self.assertEqual(MERGE_JUDGE_V1.sha256, JUDGE_V1_SHA256)  # the original judge did not change
        self.assertEqual(MERGE_JUDGE_INFORMED_V1.sha256, INFORMED_SHA256)
        self.assertEqual(MERGE_JUDGE_INFORMED_V1.prompt_id, "merge_judge_informed_v1")
        self.assertEqual(set(JUDGE_PROMPTS), {"merge_judge_v1", "merge_judge_informed_v1"})

    def test_it_differs_from_the_plain_judge_by_one_paragraph_only(self):
        v1, inf = MERGE_JUDGE_V1, MERGE_JUDGE_INFORMED_V1
        self.assertTrue(inf.system.startswith(v1.system))
        self.assertEqual(inf.system[len(v1.system):len(v1.system) + 2], "\n\n")
        self.assertEqual(inf.system[len(v1.system) + 2:].count("\n\n"), 0)  # a single paragraph
        self.assertEqual((inf.user_template, inf.output_keys, inf.examples), (v1.user_template, v1.output_keys, v1.examples))

    def test_the_paragraph_explains_but_is_not_a_hard_cannot_link_rule(self):
        paragraph = MERGE_JUDGE_INFORMED_V1.system[len(MERGE_JUDGE_V1.system):].lower()
        self.assertIn("evidence", paragraph)
        self.assertIn("weigh", paragraph)
        for hard in ("always", "never", "must", "regardless", "cannot", "do not merge", "under no", "keep_both", "required"):
            self.assertNotIn(hard, paragraph)
        # it leaves the decision with the judge and says script/spelling variants are not a difference
        self.assertIn("spelling or script variants", paragraph)

    def test_it_does_not_name_a_lexicon_term_or_a_probe_word(self):
        paragraph = MERGE_JUDGE_INFORMED_V1.system[len(MERGE_JUDGE_V1.system):].lower()
        lexicon_terms = set()
        for e in json.loads((REPO / "dkmem" / "memory" / "distinction_features.json").read_text(encoding="utf-8"))["entries"]:
            lexicon_terms |= {f.lower() for f in e["surface_forms"]}
        words = set(re.findall(r"[a-z]+", paragraph))
        self.assertEqual(lexicon_terms & words, set())
        tier1 = (REPO / "dkmem" / "data" / "tier1" / "eval_input" / "teamA_tier1_v2.jsonl").read_text(encoding="utf-8").lower()
        for term in ("devar", "jeth"):  # the illustrative terms occur nowhere in the probe input
            self.assertIn(term, paragraph)
            self.assertNotIn(term, tier1)

    def test_rendering_puts_the_guidance_in_the_system_message_only(self):
        msgs = MERGE_JUDGE_INFORMED_V1.render("Memory A: x\nMemory B: y")
        base = MERGE_JUDGE_V1.render("Memory A: x\nMemory B: y")
        self.assertEqual(msgs[1:], base[1:])
        self.assertNotEqual(msgs[0], base[0])


class TestInformedJudgeInThePipeline(unittest.TestCase):
    def run_a(self, prompt=None, modes=ALL_MODES):
        g = FakeLLM(judge=lambda a, b: ("merge", "same"))
        params = GenerationParams(seed=0)
        kw = {} if prompt is None else {"judge_prompt": prompt}
        cfg = build_run_config("C", modes, RECORDS[:3], generator=g, lexicon_path=LEXICON_PATH, params=params, **kw)
        return run_stage_attribution(RECORDS[:3], "C", load_lexicon(LEXICON_PATH), cfg, generator=g, params=params, **kw), g, cfg

    def test_default_behaviour_and_fingerprint_are_unchanged(self):
        _, _, implicit = self.run_a()
        _, _, explicit = self.run_a(MERGE_JUDGE_V1)
        self.assertEqual(implicit["fingerprint"], explicit["fingerprint"])
        self.assertEqual(implicit["prompts"]["merge_judge"], {"prompt_id": "merge_judge_v1", "sha256": JUDGE_V1_SHA256})

    def test_informed_run_is_labelled_logged_and_fingerprinted(self):
        res, g, cfg = self.run_a(MERGE_JUDGE_INFORMED_V1)
        _, _, default_cfg = self.run_a()
        self.assertNotEqual(cfg["fingerprint"], default_cfg["fingerprint"])
        self.assertEqual(cfg["prompts"]["merge_judge"], {"prompt_id": "merge_judge_informed_v1", "sha256": INFORMED_SHA256})
        self.assertTrue(all(p[0]["content"] == MERGE_JUDGE_INFORMED_V1.system for p in g.judge_prompts))
        self.assertEqual({r.strategy for r in res.rows_by_mode["off"]}, {"prompt-informed-judge"})
        self.assertEqual({r.strategy for r in res.rows_by_mode["lexicon"]}, {"dk-mem-lexicon"})
        for t in res.trace:
            self.assertEqual(t["host"]["judge"]["prompt_id"], "merge_judge_informed_v1")
            self.assertEqual(t["host"]["judge"]["prompt_sha256"], INFORMED_SHA256)
        self.assertEqual(strategy_for("C", "off", "merge_judge_informed_v1"), "prompt-informed-judge")
        self.assertEqual(strategy_for("D", "off", "merge_judge_informed_v1"), "embedding-threshold")
        self.assertEqual(strategy_for("C", "off"), "store-surface-only")

    def test_outputs_validate_and_the_manifest_carries_the_baseline_strategy(self):
        res, _, _ = self.run_a(MERGE_JUDGE_INFORMED_V1)
        with tempfile.TemporaryDirectory() as d:
            write_outputs(Path(d) / "g", res)
            man = json.loads((Path(d) / "g" / "off" / "run_manifest.json").read_text(encoding="utf-8"))
            jsonschema.validate(man, MANIFEST_SCHEMA)
            self.assertEqual((man["strategy"], man["pipeline_config"], man["dkmem_mode"]), ("prompt-informed-judge", "C", "off"))
            for line in (Path(d) / "g" / "off" / "pairwise_eval.jsonl").read_text(encoding="utf-8").splitlines():
                jsonschema.validate(json.loads(line), PAIRWISE_SCHEMA)

    def test_config_d_has_no_judge(self):
        with self.assertRaisesRegex(ValueError, "no merge judge"):
            build_run_config("D", ["off"], RECORDS[:2], generator=None, lexicon_path=LEXICON_PATH, params=GenerationParams(),
                             similarity=fake_embedding_metric(), tau=0.5, judge_prompt=MERGE_JUDGE_INFORMED_V1)

    def test_the_store_runner_takes_the_same_prompt(self):
        g = FakeLLM(judge=lambda a, b: ("merge", "same"))
        params = GenerationParams(seed=0)
        cfg = build_store_run_config("C", ["off", "lexicon"], RECORDS[:3], generator=g, lexicon_path=LEXICON_PATH, params=params,
                                     judge_prompt=MERGE_JUDGE_INFORMED_V1)
        res = run_store_attribution(RECORDS[:3], "C", load_lexicon(LEXICON_PATH), cfg, generator=g, params=params,
                                    judge_prompt=MERGE_JUDGE_INFORMED_V1)
        self.assertEqual({r.strategy for r in res.rows_by_mode["off"]}, {"prompt-informed-judge"})
        self.assertTrue(all(p[0]["content"] == MERGE_JUDGE_INFORMED_V1.system for p in g.judge_prompts))
        self.assertIn("merge_judge_informed_v1", cfg["store"]["judge"])
        # the judge host itself defaults to the plain prompt
        self.assertEqual(JudgeHost(g).prompt.prompt_id, "merge_judge_v1")

    def test_cli_flag(self):
        llm = FakeLLM()

        class Emb:
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def run_info(self):
                return {"model_config": {"model_id": "fake"}}

        with tempfile.TemporaryDirectory() as tmp:
            runs = pipeline_cli.main(
                ["--configs", "A", "C", "--modes", "off", "--judge-prompt", "merge_judge_informed_v1", "--limit", "3", "--out", tmp],
                load_generator=lambda *a: llm, load_embedder=lambda *a: Emb(),
            )
            for r in runs:
                man = json.loads((Path(r["dir"]) / "off" / "run_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(man["strategy"], "prompt-informed-judge")
            with self.assertRaises(SystemExit):
                pipeline_cli.main(["--configs", "C", "--judge-prompt", "nope", "--out", tmp], load_generator=lambda *a: llm)
            runs = store_cli.main(
                ["--configs", "C", "--modes", "off", "--judge-prompt", "merge_judge_informed_v1", "--limit", "3", "--out", tmp],
                load_generator=lambda *a: llm, load_embedder=lambda *a: Emb(),
            )
            man = json.loads((Path(runs[0]["dir"]) / "off" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(man["strategy"], "prompt-informed-judge")


# --- never-merge / flat RAG ------------------------------------------------------------------------------------


def run_flat(records=RECORDS, metric=None):
    metric = metric or fake_embedding_metric()
    cfg = build_flat_run_config(records, similarity_name=metric.name, embedder_info={"model_config": {"model_id": "fake"}})
    return cfg, run_flat_baseline(records, cfg, similarity=metric)


class TestFlatTier1(unittest.TestCase):
    def test_every_pair_is_kept_apart_so_fcr_is_zero_by_construction(self):
        cfg, res = run_flat()
        self.assertEqual(len(res.rows), len(RECORDS))
        self.assertEqual({r.decision for r in res.rows}, {"no_merge"})
        self.assertEqual({(r.strategy, r.threshold, r.compatibility, r.predicted_entity_id) for r in res.rows},
                         {("flat-dense-rag", None, None, None)})
        self.assertEqual(res.summary["unifying_decisions"], 0)
        self.assertEqual(res.summary["store"], {"entries_active": 2 * len(RECORDS), "merges": 0, "unresolved_links": 0})
        s = score([json.loads(r.to_json()) for r in res.rows], RECORDS, GOLD)["overall"]
        self.assertEqual((s["fcr"], s["mcr"]), (0.0, 1.0))  # the cost side: every same-entity pair is missed

    def test_even_identical_texts_are_never_merged(self):
        always_one = SimilarityMetric("always_one", lambda a, b: 1.0)
        _, res = run_flat(metric=always_one)
        self.assertEqual({r.decision for r in res.rows}, {"no_merge"})
        self.assertEqual({r.similarity_score for r in res.rows}, {1.0})

    def test_text_is_stored_verbatim_and_the_score_is_the_comparator(self):
        _, res = run_flat()
        r = res.rows[0]
        self.assertEqual((r.entry_a.gloss, r.entry_a.surface), (RECORDS[0].utterance_a, RECORDS[0].utterance_a))
        self.assertEqual(r.similarity_score, 0.95)
        self.assertEqual(r.record_id, f"{RECORDS[0].utterance_a_id}_e0::{RECORDS[0].utterance_b_id}_e0")

    def test_outputs_and_config(self):
        cfg, res = run_flat()
        self.assertTrue(cfg["group_id"].startswith("flat-no-llm-s0-"))
        self.assertIsNone(cfg["backbone"])
        with tempfile.TemporaryDirectory() as d:
            paths = write_flat_outputs(Path(d) / cfg["group_id"], res)
            man = json.loads(Path(paths["off/pairwise_eval"]).with_name("run_manifest.json").read_text(encoding="utf-8"))
            jsonschema.validate(man, MANIFEST_SCHEMA)
            self.assertEqual((man["strategy"], man["backbone"], man["dkmem_mode"]), ("flat-dense-rag", None, "off"))
            for line in Path(paths["off/pairwise_eval"]).read_text(encoding="utf-8").splitlines():
                jsonschema.validate(json.loads(line), PAIRWISE_SCHEMA)
            self.assertTrue(Path(paths["off/store_events"]).exists() and Path(paths["off/store_final"]).exists())
            final = [json.loads(l) for l in Path(paths["off/store_final"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual({len(f["entries"]) for f in final}, {2})

    def test_unsupported_language_is_skipped_and_duplicates_rejected(self):
        recs = [RECORDS[0], rec("ep_x", "a", "b", lang="x")]
        _, res = run_flat(recs)
        self.assertEqual(len(res.rows), 1)
        self.assertEqual(res.summary["episodes_skipped_unsupported_language"], 1)
        with self.assertRaises(ValueError):
            run_flat([RECORDS[0], RECORDS[0]])


# --- Tier 2 retrieval / QA on the shared infrastructure -----------------------------------------------------------


class TestFlatTier2(unittest.TestCase):
    TURNS = [
        Turn("c1_s1_t1", "meri chachi Pune mein rehti hai", "hi", "s1", 1),
        Turn("c1_s1_t2", "mera bhai Delhi mein kaam karta hai", "hi", "s1", 2),
        Turn("c1_s2_t1", "meri mausi Delhi mein rehti hai", "hi", "s2", 1),
        Turn("c1_s2_t2", "main cricket khelta hoon", "hi", "s2", 2),
    ]

    @staticmethod
    def metric():
        def sim(a, b):
            wa, wb = set(a.casefold().split()), set(b.casefold().split())
            return len(wa & wb) / len(wa | wb) if wa | wb else 0.0
        return SimilarityMetric("token_jaccard", sim)

    def rag(self):
        rag = FlatRAG("c1", self.metric())
        rag.ingest(self.TURNS)
        return rag

    def test_every_turn_is_kept_even_near_duplicates(self):
        rag = self.rag()
        rag.ingest([Turn("c1_s3_t1", "meri chachi Pune mein rehti hai", "hi", "s3", 1)])  # a verbatim repeat
        self.assertEqual(len(rag.store.active()), 5)
        self.assertEqual(rag.store.stats()["entries_absorbed"], 0)
        self.assertEqual(len(rag.store.links), 0)
        self.assertEqual(rag.store.get("c1_s1_t1_e0").session_id, "s1")

    def test_retrieval_uses_the_shared_retrieve_and_ranks_by_the_comparator(self):
        rag = self.rag()
        top = rag.retrieve("mausi Delhi mein", 2)
        self.assertEqual(rag.store.get(top[0].entry_id).source_utterance_id, "c1_s2_t1")
        self.assertEqual(len(top), 2)

    def test_retrieval_report(self):
        rag = self.rag()
        qs = [Question("q1", "mausi kahan rehti hai", "Delhi", ("c1_s2_t1",)),
              Question("q2", "chachi kahan rehti hai", "Pune", ("c1_s1_t1",)),
              Question("q3", "bhai aur chachi", "-", ("c1_s1_t2", "c1_s1_t1"))]
        rep = retrieval_report(rag, qs, ks=(1, 4))
        self.assertEqual(rep["n_questions"], 3)
        self.assertEqual(rep["by_k"]["4"]["all_evidence"], 1.0)  # k covers the whole store
        self.assertLess(rep["by_k"]["1"]["all_evidence"], 1.0)   # q3 needs two utterances

    def test_answering_uses_the_retrieved_memories_and_scores_the_answer(self):
        rag = self.rag()
        seen = []

        class Gen:
            backbone = "fake"

            def generate(self, prompts, params=None):
                seen.extend(prompts)
                return ["Delhi.", "I think Mumbai"]

        qs = [Question("q1", "mausi kahan rehti hai", "Delhi", ("c1_s2_t1",)),
              Question("q2", "chachi kahan rehti hai", "Pune", ("c1_s1_t1",))]
        res = answer_questions(rag, qs, Gen(), k=2)
        self.assertEqual([r.contains_gold for r in res], [True, False])
        self.assertTrue(res[0].evidence_found)
        self.assertEqual(seen[0][0]["content"], FLAT_QA_V1.system)
        self.assertIn("- meri mausi Delhi mein rehti hai", seen[0][-1]["content"])
        self.assertTrue(seen[0][-1]["content"].endswith("Question: mausi kahan rehti hai"))
        rep = qa_report(res)
        self.assertEqual((rep["n"], rep["contains_gold"]), (2, 0.5))
        self.assertEqual(rep["prompt_sha256"], FLAT_QA_V1.sha256)

    def test_answer_metrics(self):
        self.assertEqual(normalize_answer(" Delhi,  India! "), "delhi india")
        self.assertEqual(token_f1("Delhi", "delhi"), 1.0)
        self.assertAlmostEqual(token_f1("in Delhi city", "Delhi"), 0.5)
        self.assertEqual(token_f1("Pune", "Delhi"), 0.0)

    def test_the_same_ingestion_path_runs_a_consolidating_host(self):
        from dkmem.store.store import MemoryStore

        store = MemoryStore("c2")
        ingest_conversation(store, self.TURNS, ThresholdHost(0.2), self.metric(), ConsolidationConfig("off", 5))
        self.assertLess(len(store.active()), len(self.TURNS))  # merges happened: this is not the flat host
        flat = MemoryStore("c3")
        ingest_conversation(flat, self.TURNS, NeverMergeHost(), self.metric(), ConsolidationConfig("off", 5))
        self.assertEqual(len(flat.active()), len(self.TURNS))
        with self.assertRaises(ValueError):
            ingest_conversation(MemoryStore("c4"), self.TURNS, NeverMergeHost(), self.metric(), ConsolidationConfig("lexicon", 5))


# --- the baselines CLI -----------------------------------------------------------------------------------------------


class FakeEmbedder:
    def embed(self, texts):
        return hashing_embed(texts)

    def run_info(self):
        return {"model_config": {"model_id": "fake-embedder", "revision": None, "dtype": "float16"}, "resolved_revision": "r1"}


class TestBaselinesCli(unittest.TestCase):
    def test_flat_and_mem0_commands_write_groups_in_the_ad_layout(self):
        calls = {}

        def lg(model_id, revision, device, shard):
            calls["generator"] = (model_id, revision, device, shard)
            return FakeMem0LLM(facts=facts_fn)

        with tempfile.TemporaryDirectory() as tmp:
            flat = baselines_cli.main(["flat", "--limit", "4", "--out", tmp], load_generator=lg, load_embedder=lambda *a: FakeEmbedder())
            mem0 = baselines_cli.main(["mem0", "--limit", "4", "--model-id", "fake/qwen", "--revision", "abc", "--out", tmp],
                                      load_generator=lg, load_embedder=lambda *a: FakeEmbedder())
            self.assertEqual(calls["generator"], ("fake/qwen", "abc", "cuda:0", False))
            for out in (flat, mem0):
                d = Path(out["dir"])
                for name in ("run_config.json", "trace.jsonl", "summary.json", "invocation.json"):
                    self.assertTrue((d / name).exists(), name)
                cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual(cfg["input"]["limit"], 4)
                self.assertTrue(cfg["input"]["sanctioned"])
                self.assertEqual(len(cfg["code"]["tree_sha256"]), 64)
                for line in (d / "off" / "pairwise_eval.jsonl").read_text(encoding="utf-8").splitlines():
                    jsonschema.validate(json.loads(line), PAIRWISE_SCHEMA)
            self.assertEqual(flat["summary"]["unifying_decisions"], 0)
            self.assertEqual(json.loads((Path(mem0["dir"]) / "run_config.json").read_text(encoding="utf-8"))["mem0"]["commit"],
                             "144627c4ce5bc4db6acac17cbd158065f2b27a8d")


# --- the same experiment through every system ------------------------------------------------------------------------


class TestCrossSystemFairness(unittest.TestCase):
    def test_a_d_mem0_and_flat_groups_on_one_input_are_comparable_and_scored_by_one_function(self):
        params = GenerationParams(seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeMem0LLM(facts=facts_fn)
            input_info = {"file_sha256": "f", "sanctioned": False}
            mem_cfg = build_mem0_run_config(RECORDS, generator=llm, params=params, input_info=input_info,
                                            embedder_info={"model_config": {"model_id": "fake"}})
            mem0 = run_mem0_baseline(RECORDS, mem_cfg, generator=llm, params=params, embed_texts=hashing_embed)
            write_mem0_outputs(Path(tmp) / "mem0", mem0)
            flat_cfg, flat = run_flat()
            write_flat_outputs(Path(tmp) / "flat", flat)
            d = run_d(0.8)
            write_outputs(Path(tmp) / "d", d)
            groups = [load_group(Path(tmp) / n) for n in ("mem0", "flat", "d")]
            check = check_comparable(groups)
            # inputs and pair order agree across all three; the fake generators differ in backbone/embedder names by
            # construction, which is exactly what the check is for -- so only those keys may be reported
            self.assertEqual({i["key"] for i in check["issues"]} - {"inputs", "embedder", "backbone", "decoding"}, set())
            self.assertNotIn("pair_order", {i["key"] for i in check["issues"]})
            for g in groups:
                s = score(g.rows(g.modes[0]), RECORDS, GOLD, no_entry_ids=g.no_entry_ids())["overall"]
                self.assertEqual(s["n_different"] + s["n_same"], len(GOLD))
            # the flat anchor is the FCR floor of the set
            fcrs = {g.strategy_label: score(g.rows("off"), RECORDS, GOLD)["overall"]["fcr"] for g in groups}
            self.assertEqual(fcrs["flat-dense-rag"], 0.0)


if __name__ == "__main__":
    unittest.main()
