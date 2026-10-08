"""Tests for the end-to-end store runner (dkmem.store.runner / dkmem.store.cli).

No GPU: a scripted fake LLM answers the V4, native and judge prompts and a fake metric/embedder
stands in for bge-m3 (the same fakes the A-D harness tests use). The central check is
*equivalence with the frozen harness*: the store runner and ``run_stage_attribution`` are run on
the same records with the same fake model, and every pairwise row, every trace field the two
share, and the shared summary counts must agree -- including failures (a V4 / native extraction
failure, a judge failure, an unsupported language) and the ``lexicon+llm`` pair that has no V4
extraction.

Run: python -m unittest discover -s tests   (test_pipeline's FakeLLM is imported by name)
"""

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema
from test_pipeline import LEXICON_PATH, FakeLLM, rec

from dkmem.backends.llm import GenerationParams
from dkmem.memory.cache import ExtractionCache
from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.similarity import DEFAULT_SIMILARITY, SimilarityMetric
from dkmem.pipeline import cli as pipeline_cli
from dkmem.pipeline.runner import build_run_config, run_stage_attribution
from dkmem.store import cli as store_cli
from dkmem.store.consolidate import CachingHost, HostProposal, HostQuery
from dkmem.store.runner import (
    STORE_RUN_CONFIG_SCHEMA,
    STORE_TRACE_SCHEMA,
    build_store_run_config,
    run_store_attribution,
    write_store_outputs,
)

REPO = Path(__file__).parent.parent
PAIRWISE_SCHEMA = json.loads((REPO / "dkmem" / "pairwise_eval.schema.json").read_text(encoding="utf-8"))
MANIFEST_SCHEMA = json.loads((REPO / "dkmem" / "run_manifest.schema.json").read_text(encoding="utf-8"))
ALL_MODES = ["off", "lexicon", "lexicon+llm"]


class ScriptedLLM(FakeLLM):
    """FakeLLM whose V4 and native extractions fail for utterances containing ``FAIL``."""

    def generate(self, prompts, params=None):
        out = super().generate(prompts, params)
        for i, p in enumerate(prompts):
            system, last = p[0]["content"], p[-1]["content"]
            is_extraction = 'exactly one key and nothing else' in system or system.startswith("You convert one user utterance")
            if is_extraction and "FAIL" in last:
                out[i] = "this is not json"
        return out


def llm():
    def judge(a, b):
        return "maybe?" if "BADJUDGE" in a else ("merge", "same")

    return ScriptedLLM(
        gloss=lambda u: u,  # the gloss is the utterance: the judge then sees distinguishable texts
        model_distinction=lambda u: {"kinship": "bhaiya"} if "bhaiya" in u else {},
        judge=judge,
    )


RECORDS = [
    rec("ep_1", "meri chachi Pune mein rehti hai", "meri mausi Pune mein rehti hai"),        # incompatible
    rec("ep_2", "mera bhai Delhi gaya", "mera bhai Delhi gaya tha"),                         # same entity
    rec("ep_3", "FAIL meri chachi Pune mein hai", "meri mausi Pune mein hai"),               # extraction fails (A, B); V4 fails (lexicon+llm)
    rec("ep_4", "BADJUDGE meri bua Pune mein hai", "meri mami Pune mein hai"),              # judge fails
    rec("ep_5", "meri chachi Pune mein hai", "meri mausi Pune mein hai", lang="x"),           # unsupported language
    rec("ep_6", "my aunt lives in Pune", "my aunt lives in Delhi", lang="en"),
    rec("ep_7", "mera bhaiya Pune gaya", "mera bhai Pune gaya"),                              # model-only distinction -> underdetermined
    rec("ep_8", "meri chachi Pune mein rehti hai", "meri chachi Delhi mein rehti hai"),       # compatible (same kinship)
]

SHARED_TRACE_KEYS = ("status", "eval_pair_id", "language", "record_id", "input", "extraction", "stored",
                     "similarity", "host", "gate", "final", "diagnosis")


def fake_metric():
    return SimilarityMetric("fake_embedding", lambda a, b: min(1.0, DEFAULT_SIMILARITY(a, b) + 0.1))


def run_both(config_id, records=RECORDS, *, modes=ALL_MODES, tau=None, make_llm=llm, **store_kwargs):
    lexicon = load_lexicon(LEXICON_PATH)
    params = GenerationParams(seed=0)
    sim = fake_metric() if config_id == "D" else None
    generator = lambda: make_llm()  # noqa: E731
    g1, g2 = generator(), generator()
    cfg1 = build_run_config(config_id, modes, records, generator=g1, lexicon_path=LEXICON_PATH, params=params, similarity=sim, tau=tau)
    frozen = run_stage_attribution(records, config_id, lexicon, cfg1, generator=g1, params=params, similarity=sim, tau=tau)
    cfg2 = build_store_run_config(config_id, modes, records, generator=g2, lexicon_path=LEXICON_PATH, params=params, similarity=sim, tau=tau,
                                  **{k: v for k, v in store_kwargs.items() if k == "k"})
    store = run_store_attribution(records, config_id, lexicon, cfg2, generator=g2, params=params, similarity=sim, tau=tau, **store_kwargs)
    return frozen, store, g1, g2


def without_run_id(row):
    return {k: v for k, v in row.items() if k != "run_id"}


class TestEquivalenceWithTheFrozenHarness(unittest.TestCase):
    def check(self, config_id, tau=None, modes=ALL_MODES):
        frozen, store, _g1, _g2 = run_both(config_id, tau=tau, modes=modes)
        for mode in modes:
            with self.subTest(config=config_id, tau=tau, mode=mode):
                a = [without_run_id(json.loads(r.to_json())) for r in frozen.rows_by_mode[mode]]
                b = [without_run_id(json.loads(r.to_json())) for r in store.rows_by_mode[mode]]
                self.assertEqual(b, a)
                self.assertEqual({r.run_id for r in store.rows_by_mode[mode]} | set(), {f"{store.run_config['group_id']}-{mode.replace('+', '-')}"} if b else set())
        by_id = {t["eval_pair_id"]: t for t in store.trace}
        self.assertEqual([t["eval_pair_id"] for t in store.trace], [r.eval_pair_id for r in RECORDS])
        for t in frozen.trace:
            mine = by_id[t["eval_pair_id"]]
            for key in SHARED_TRACE_KEYS:
                with self.subTest(config=config_id, tau=tau, pair=t["eval_pair_id"], key=key):
                    self.assertEqual(mine.get(key), t.get(key))
        for key, value in frozen.summary.items():
            with self.subTest(config=config_id, summary=key):
                self.assertEqual(store.summary[key], value)
        return frozen, store

    def test_config_a(self):
        frozen, store = self.check("A")
        self.assertEqual(store.summary["sides_extraction_failed"], 1)
        self.assertEqual(by_status(store.trace)["ep_3"], "extraction_failed")

    def test_config_b(self):
        _, store = self.check("B")
        self.assertEqual(by_status(store.trace)["ep_3"], "extraction_failed")
        self.assertIn("stored_language", store.summary)

    def test_config_c(self):
        _, store = self.check("C")
        # the V4 call fails for ep_3's side a: the pair is judged and kept in off / lexicon but has no row in lexicon+llm
        self.assertEqual(by_status(store.trace)["ep_3"], "ok")
        ids = {m: {r.record_id for r in store.rows_by_mode[m]} for m in ALL_MODES}
        self.assertIn("ep_3_utt_a_e0::ep_3_utt_b_e0", ids["off"])
        self.assertIn("ep_3_utt_a_e0::ep_3_utt_b_e0", ids["lexicon"])
        self.assertNotIn("ep_3_utt_a_e0::ep_3_utt_b_e0", ids["lexicon+llm"])

    def test_config_d_at_three_taus(self):
        for tau in (0.5, 0.8, 0.9):
            self.check("D", tau=tau)

    def test_the_comparison_is_not_vacuous(self):
        _, store = self.check("C")
        rows = {m: store.rows_by_mode[m] for m in ALL_MODES}
        self.assertIn("incompatible", {r.compatibility for r in rows["lexicon"]})
        self.assertIn("underdetermined", {r.compatibility for r in rows["lexicon+llm"]})
        self.assertIn("no_merge", {r.decision for r in rows["lexicon"]})
        self.assertIn("underdetermined_link", {r.decision for r in rows["lexicon+llm"]})
        self.assertIn("merge", {r.decision for r in rows["off"]})
        self.assertTrue(any(t["status"] == "judge_failed" for t in store.trace))
        self.assertTrue(any(t["status"] == "unsupported_language" for t in store.trace))


def by_status(trace):
    return {t["eval_pair_id"]: t["status"] for t in trace}


class TestStoreSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _f, cls.res, cls.llm_frozen, cls.llm_store = run_both("C")

    def test_one_fresh_store_per_episode_and_mode(self):
        supported = [r.eval_pair_id for r in RECORDS if r.language != "x"]
        for mode in ALL_MODES:
            stores = self.res.stores_by_mode[mode]
            self.assertEqual([s.conv_id for s in stores], supported)
            for s in stores:
                self.assertLessEqual(len(s.entries), 2)
                self.assertTrue(all(e.conv_id == s.conv_id for e in s.entries.values()))
                self.assertTrue(all(e.source_utterance_id.startswith(s.conv_id) for e in s.entries.values()))
                self.assertEqual(s.integrity_errors(), [])
                if mode != "off":
                    self.assertEqual(s.gate_conflicts(), [])

    def test_off_merges_across_a_distinction_the_gate_keeps_apart(self):
        by = {m: {s.conv_id: s for s in self.res.stores_by_mode[m]} for m in ALL_MODES}
        self.assertEqual(len(by["off"]["ep_1"].active()), 1)       # chachi + mausi merged
        self.assertEqual(len(by["lexicon"]["ep_1"].active()), 2)
        self.assertEqual(len(by["lexicon+llm"]["ep_1"].active()), 2)
        merged = by["off"]["ep_1"].active()[0]
        self.assertEqual(merged.stored_text, "meri mausi Pune mein rehti hai")  # newer text wins
        self.assertEqual(merged.history[0]["replaced_text"], "meri chachi Pune mein rehti hai")
        self.assertEqual(len(by["lexicon+llm"]["ep_7"].links), 1)  # underdetermined -> logged link
        self.assertEqual(len(by["lexicon"]["ep_7"].links), 0)

    def test_no_v4_fallback_the_write_is_skipped_and_logged(self):
        store = next(s for s in self.res.stores_by_mode["lexicon+llm"] if s.conv_id == "ep_3")
        kinds = [ev["event"] for ev in store.events]
        self.assertEqual(kinds.count("write_skipped"), 1)
        skipped = next(ev for ev in store.events if ev["event"] == "write_skipped")
        self.assertEqual(skipped["entry"]["entry_id"], "ep_3_utt_a_e0")
        self.assertIn("not written", skipped["reason"])
        self.assertNotIn("ep_3_utt_a_e0", store.entries)
        row = next(t for t in self.res.trace if t["eval_pair_id"] == "ep_3")
        self.assertEqual(row["gate"]["lexicon+llm"], {"applied": False, "skipped": "no V4 extraction for the model fallback"})
        self.assertIsNone(row["final"]["lexicon+llm"])
        self.assertEqual(row["store"]["lexicon+llm"]["actions"], ["skipped", "add"])
        self.assertEqual(row["store"]["lexicon"]["actions"], ["add", "add"])

    def test_judge_failure_is_logged_not_decided(self):
        row = next(t for t in self.res.trace if t["eval_pair_id"] == "ep_4")
        self.assertEqual(row["status"], "judge_failed")
        self.assertIsNone(row["final"])
        self.assertIsNotNone(row["host"]["judge"]["error"])
        for mode in ALL_MODES:
            self.assertNotIn("ep_4_utt_a_e0::ep_4_utt_b_e0", {r.record_id for r in self.res.rows_by_mode[mode]})
        store = next(s for s in self.res.stores_by_mode["off"] if s.conv_id == "ep_4")
        self.assertTrue(any(ev.get("had_failures") for ev in store.events))
        self.assertEqual(self.res.summary["pairs_judge_failed"], 1)
        self.assertEqual(self.res.summary["store"]["off"]["host_failures"], 1)

    def test_the_judge_runs_once_per_pair_and_is_shared_across_modes(self):
        pairs = sum(1 for t in self.res.trace if t["status"] in ("ok", "judge_failed"))
        self.assertEqual(self.llm_store.calls["judge"], pairs)
        self.assertEqual(self.llm_frozen.calls["judge"], pairs)
        stats = self.res.summary["host_cache"]
        self.assertEqual((stats["prewarmed"], stats["live"]), (pairs, 0))
        self.assertGreaterEqual(stats["reused"], pairs * 2)

    def test_summary_store_counts(self):
        s = self.res.summary["store"]["off"]
        self.assertEqual(s["episodes"], 7)
        self.assertEqual(s["writes"], 14)
        self.assertEqual(s["adds"] + s["merges"] + s["writes_skipped"], s["writes"])
        self.assertEqual(self.res.summary["store"]["lexicon+llm"]["writes_skipped"], 1)
        self.assertEqual(self.res.summary["store"]["lexicon+llm"]["unresolved_links"], 1)

    def test_trace_schema_and_order(self):
        self.assertEqual({t["schema"] for t in self.res.trace}, {STORE_TRACE_SCHEMA})
        self.assertEqual(self.res.run_config["schema"], STORE_RUN_CONFIG_SCHEMA)
        self.assertTrue(self.res.run_config["group_id"].startswith("store-C-"))


class TestRunConfig(unittest.TestCase):
    def cfg(self, **kw):
        base = dict(generator=llm(), lexicon_path=LEXICON_PATH, params=GenerationParams(seed=0))
        base.update(kw)
        return build_store_run_config("C", ALL_MODES, RECORDS, **base)

    def test_reproducible_fingerprint_and_store_section(self):
        a, b = self.cfg(), self.cfg()
        self.assertEqual(a["fingerprint"], b["fingerprint"])
        self.assertEqual(a["group_id"], b["group_id"])
        self.assertEqual(a["store"]["k"], 5)
        self.assertEqual(a["store"]["discriminative_features"], ["kinship", "register"])
        self.assertEqual(a["tau"], None)
        self.assertEqual(len(a["code"]["tree_sha256"]), 64)
        self.assertIn("store/runner.py", a["code"]["files"])
        for key in ("seed", "backbone", "generation_params", "prompts", "input", "lexicon", "pipeline_config"):
            self.assertIn(key, a)

    def test_every_input_that_decides_the_output_changes_the_fingerprint(self):
        base = self.cfg()["fingerprint"]
        self.assertNotEqual(self.cfg(k=3)["fingerprint"], base)
        self.assertNotEqual(self.cfg(params=GenerationParams(seed=1))["fingerprint"], base)
        self.assertNotEqual(build_store_run_config("C", ["off"], RECORDS, generator=llm(), lexicon_path=LEXICON_PATH,
                                                   params=GenerationParams(seed=0))["fingerprint"], base)
        self.assertNotEqual(build_store_run_config("C", ALL_MODES, RECORDS[:3], generator=llm(), lexicon_path=LEXICON_PATH,
                                                   params=GenerationParams(seed=0))["fingerprint"], base)

    def test_not_the_same_group_as_the_frozen_harness(self):
        frozen = build_run_config("C", ALL_MODES, RECORDS, generator=llm(), lexicon_path=LEXICON_PATH, params=GenerationParams(seed=0))
        self.assertNotEqual(frozen["group_id"], self.cfg()["group_id"])
        self.assertNotEqual(frozen["fingerprint"], self.cfg()["fingerprint"])


class TestRunnerValidation(unittest.TestCase):
    def run_it(self, config_id, records=RECORDS, generator="llm", **kw):
        generator = llm() if generator == "llm" else generator
        lexicon = load_lexicon(LEXICON_PATH)
        params = GenerationParams(seed=0)
        modes = kw.pop("modes", ["off"])
        cfg = build_store_run_config(config_id, modes, records, generator=generator, lexicon_path=LEXICON_PATH, params=params,
                                     similarity=kw.get("similarity"), tau=kw.get("tau"))
        return run_store_attribution(records, config_id, lexicon, cfg, generator=generator, params=params, **kw)

    def test_config_d_needs_metric_and_tau_and_judge_configs_take_no_tau(self):
        for kw in ({}, {"tau": 0.8}, {"similarity": fake_metric()}):
            with self.assertRaises(ValueError):
                self.run_it("D", **kw)
        with self.assertRaises(ValueError):
            build_store_run_config("C", ["off"], RECORDS, generator=llm(), lexicon_path=LEXICON_PATH,
                                   params=GenerationParams(seed=0), tau=0.8)

    def test_generator_required_for_judge_configs_and_for_lexicon_llm(self):
        with self.assertRaises(ValueError):
            self.run_it("C", generator=None)
        with self.assertRaises(ValueError):
            self.run_it("D", generator=None, modes=["lexicon+llm"], similarity=fake_metric(), tau=0.8)
        res = self.run_it("D", generator=None, modes=["off", "lexicon"], similarity=fake_metric(), tau=0.8)
        self.assertEqual(len(res.rows_by_mode["off"]), 7)

    def test_duplicate_episode_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.run_it("C", records=[rec("ep_1"), rec("ep_1")])


class TestExtractionCache(unittest.TestCase):
    def test_second_run_generates_no_v4_and_gives_the_same_rows(self):
        records = [RECORDS[0], RECORDS[1], RECORDS[5], RECORDS[7]]  # no failing utterances
        lexicon = load_lexicon(LEXICON_PATH)
        params = GenerationParams(seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            results = []
            for _ in range(2):
                g = llm()
                cfg = build_store_run_config("A", ALL_MODES, records, generator=g, lexicon_path=LEXICON_PATH, params=params)
                res = run_store_attribution(records, "A", lexicon, cfg, generator=g, params=params, extraction_cache=ExtractionCache(path))
                results.append((res, g.calls["v4"]))
            (r1, v1), (r2, v2) = results
            self.assertEqual((v1, v2), (8, 0))
            self.assertEqual(r1.summary["extraction_cache"], {"hits": 0, "generated": 8})
            self.assertEqual(r2.summary["extraction_cache"], {"hits": 8, "generated": 0})
            rows = lambda r: [[without_run_id(json.loads(x.to_json())) for x in r.rows_by_mode[m]] for m in ALL_MODES]  # noqa: E731
            self.assertEqual(rows(r1), rows(r2))
            # and the cached run equals the uncached frozen harness
            frozen, _s, _g1, _g2 = run_both("A", records)
            self.assertEqual(rows(r2), [[without_run_id(json.loads(x.to_json())) for x in frozen.rows_by_mode[m]] for m in ALL_MODES])


class TestCachingHost(unittest.TestCase):
    class Counting:
        mechanism = "counting"

        def __init__(self):
            self.calls = []

        def propose(self, queries):
            self.calls.append([(q.candidate_text, q.new_text) for q in queries])
            return [HostProposal(None if q.new_text == "bad" else "merge", {"n": len(self.calls)}) for q in queries]

    def test_prewarm_batches_once_and_later_queries_are_free(self):
        inner = self.Counting()
        h = CachingHost(inner)
        h.prewarm([("a", "b"), ("c", "bad"), ("a", "b")])
        self.assertEqual(inner.calls, [[("a", "b"), ("c", "bad")]])
        out = h.propose([HostQuery("x", "a", "b", 0.1), HostQuery("y", "c", "bad", 0.2)])
        self.assertEqual([p.decision for p in out], ["merge", None])  # a failed answer is cached too, not retried
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(h.stats, {"live": 0, "reused": 2, "prewarmed": 2})
        h.propose([HostQuery("x", "new", "pair", 0.3)])
        self.assertEqual((len(inner.calls), h.stats["live"]), (2, 1))
        self.assertEqual(h.propose([]), [])


class TestOutputsAndCli(unittest.TestCase):
    def loaders(self):
        fake = llm()
        calls = {}

        def load_generator(model_id, revision, device, shard):
            calls["generator"] = (model_id, revision, device, shard)
            return fake

        class FakeEmbedder:
            def embed(self, texts):
                import math
                return [[math.cos(math.radians(sum(map(ord, t)) % 90)), math.sin(math.radians(sum(map(ord, t)) % 90))] for t in texts]

            def run_info(self):
                return {"model_config": {"model_id": "fake-embedder", "revision": "r"}, "resolved_revision": "r"}

        def load_embedder(model_id, revision, device):
            calls["embedder"] = (model_id, revision, device)
            return FakeEmbedder()

        return calls, load_generator, load_embedder

    def test_cli_runs_a_to_d_on_real_input_and_matches_the_frozen_cli(self):
        argv = ["--configs", "A", "B", "C", "D", "--modes", "off", "lexicon", "lexicon+llm", "--model-id", "fake/qwen",
                "--seed", "3", "--tau-d", "0.5", "0.9999", "--limit", "10", "--embedding-revision", "emb-rev"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            calls, lg, le = self.loaders()
            runs = store_cli.main([*argv, "--out", str(tmp / "store")], load_generator=lg, load_embedder=le)
            calls2, lg2, le2 = self.loaders()
            frozen = pipeline_cli.main([*argv, "--out", str(tmp / "frozen")], load_generator=lg2, load_embedder=le2)
            self.assertEqual([(r["config"], r["tau"]) for r in runs], [("A", None), ("B", None), ("C", None), ("D", 0.5), ("D", 0.9999)])
            self.assertEqual(calls["embedder"], ("BAAI/bge-m3", "emb-rev", "cuda:0"))
            self.assertEqual(len({r["group_id"] for r in runs}), 5)
            for r, f in zip(runs, frozen):
                d, fd = Path(r["dir"]), Path(f["dir"])
                self.assertTrue(r["group_id"].startswith("store-"))
                for name in ("run_config.json", "trace.jsonl", "summary.json", "invocation.json"):
                    self.assertTrue((d / name).exists(), name)
                cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual((cfg["seed"], cfg["tau"], cfg["input"]["limit"], cfg["input"]["sanctioned"]), (3, r["tau"], 10, True))
                self.assertEqual(cfg["input"]["file_sha256"], "eab5dcb23375eb2e744c72a98cb31c25da0171b2ba3297a0843574ed9a34cd07")
                self.assertEqual(cfg["store"]["k"], 5)
                trace = [json.loads(l) for l in (d / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(trace), 10)
                for mode_dir in ("off", "lexicon", "lexicon-llm"):
                    man = json.loads((d / mode_dir / "run_manifest.json").read_text(encoding="utf-8"))
                    jsonschema.validate(man, MANIFEST_SCHEMA)
                    rows = [json.loads(l) for l in (d / mode_dir / "pairwise_eval.jsonl").read_text(encoding="utf-8").splitlines()]
                    frozen_rows = [json.loads(l) for l in (fd / mode_dir / "pairwise_eval.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(man["run_id"], f"{r['group_id']}-{mode_dir}")
                    self.assertEqual({row["run_id"] for row in rows} | {man["run_id"]}, {man["run_id"]})
                    for row in rows:
                        jsonschema.validate(row, PAIRWISE_SCHEMA)
                        self.assertEqual(row["threshold"], r["tau"])
                    self.assertEqual([without_run_id(x) for x in rows], [without_run_id(x) for x in frozen_rows])
                    events = [json.loads(l) for l in (d / mode_dir / "store_events.jsonl").read_text(encoding="utf-8").splitlines()]
                    finals = [json.loads(l) for l in (d / mode_dir / "store_final.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(len(finals), 10)
                    self.assertEqual(len({fin["conv_id"] for fin in finals}), 10)
                    self.assertEqual(sum(1 for ev in events if ev["event"] in ("write", "write_skipped")), 20)
                    self.assertTrue(all("conv_id" in ev for ev in events))
            d_runs = [r for r in runs if r["config"] == "D"]
            self.assertNotEqual(d_runs[0]["summary"]["final_decisions"], d_runs[1]["summary"]["final_decisions"])

    def test_cli_flags_and_validation(self):
        calls, lg, le = self.loaders()
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "x.jsonl"
            runs = store_cli.main(["--configs", "A", "--modes", "off", "--limit", "2", "--k", "3", "--extraction-cache", str(cache),
                                   "--model-id", "fake/qwen", "--out", tmp], load_generator=lg, load_embedder=le)
            cfg = json.loads((Path(runs[0]["dir"]) / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["store"]["k"], 3)
            self.assertTrue(cache.exists())
            for argv in (["--configs", "D", "--out", tmp], ["--configs", "A", "--tau-d", "0.5", "--out", tmp],
                         ["--configs", "D", "--tau-d", "1.5", "--out", tmp], ["--configs", "A", "--k", "0", "--out", tmp]):
                with self.assertRaises(SystemExit):
                    store_cli.main(argv, load_generator=lg, load_embedder=le)

    def test_config_d_without_llm_modes_loads_no_generator(self):
        calls, lg, le = self.loaders()
        with tempfile.TemporaryDirectory() as tmp:
            store_cli.main(["--configs", "D", "--modes", "off", "lexicon", "--tau-d", "0.8", "--limit", "2", "--out", tmp],
                           load_generator=lg, load_embedder=le)
        self.assertNotIn("generator", calls)

    def test_write_store_outputs_returns_every_path(self):
        frozen, res, _g1, _g2 = run_both("D", tau=0.8)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_store_outputs(tmp, res)
        for m in ALL_MODES:
            self.assertIn(f"{m}/store_events", paths)
            self.assertIn(f"{m}/store_final", paths)
            self.assertIn(f"{m}/pairwise_eval", paths)
        self.assertIn("trace", paths)


if __name__ == "__main__":
    unittest.main()
