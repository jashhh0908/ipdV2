"""Tests for the offline evaluation (dkmem.eval): FCR/MCR per the Tier 1 contract, the exact raise-tau sweep, the
DK-Mem ablation report, the fairness check and the CLI.

The sweep is validated against ground truth: Config D is *actually run* at several taus with the real harness, and the
sweep computed from a single run's log must reproduce each of those runs' FCR/MCR exactly, in every DK-Mem mode.

Run: python -m unittest discover -s tests   (test_pipeline's helpers are imported by name)
"""

import json
import tempfile
import unittest
from pathlib import Path

from test_pipeline import LEXICON_PATH, FakeLLM, rec

from dkmem.backends.llm import GenerationParams
from dkmem.eval import cli as eval_cli
from dkmem.eval.ablation import ablation_report
from dkmem.eval.fairness import check_comparable
from dkmem.eval.gold import GoldError, GoldPair, gold_from_pairs, load_gold
from dkmem.eval.groups import load_group
from dkmem.eval.metrics import INDICATIVE_BELOW, pair_outcomes, rates, score
from dkmem.eval.sweep import (
    SweepError,
    best_fcr_at_mcr,
    compare_at_matched_mcr,
    final_decision_at,
    pareto_frontier,
    sweep_config_d,
    tau_grid,
)
from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.similarity import SimilarityMetric
from dkmem.pipeline.runner import build_run_config, run_stage_attribution, write_outputs

ALL_MODES = ["off", "lexicon", "lexicon+llm"]


# --- synthetic data ---------------------------------------------------------------------------------------------

PAIRS = [
    # id, utterance a, utterance b, gold relation, class, cosine score the fake embedder gives the pair
    ("ep_1", "meri chachi Pune mein rehti hai", "meri mausi Pune mein rehti hai", "different", "kinship", 0.95),
    ("ep_2", "meri chachi Pune mein rehti hai", "meri chachi Delhi mein rehti hai", "same", "kinship", 0.90),
    ("ep_3", "mera bhai Delhi gaya", "mera bhai Delhi gaya tha", "same", "name_variant", 0.80),
    ("ep_4", "meri bua Pune mein hai", "meri mami Pune mein hai", "different", "kinship", 0.85),
    ("ep_5", "tu kahan hai", "aap kahan hai", "different", "honorific_register", 0.70),
    ("ep_6", "kal baarish hui", "voh school gaya", "different", "name_variant", 0.60),
    ("ep_7", "mera naam Neha hai", "mera naam Neha hai ji", "same", "name_variant", 0.92),
    ("ep_8", "meri mausi Pune mein hai", "woh Pune mein hai", "different", "kinship", 0.88),
    ("ep_9", "meri chachi aur mausi Pune gayi", "meri chachi Pune gayi", "same", "kinship", 0.75),  # a is lexicon-ambiguous
]
RECORDS = [rec(p, a, b) for p, a, b, *_ in PAIRS]
GOLD = gold_from_pairs(GoldPair(p, rel, cls, "hi") for p, _a, _b, rel, cls, _s in PAIRS)
SCORES = {(a, b): s for _p, a, b, _r, _c, s in PAIRS}


def fake_embedding_metric():
    return SimilarityMetric("fake_embedding", lambda a, b: SCORES.get((a, b), SCORES.get((b, a))))


def llm():
    return FakeLLM(model_distinction=lambda u: {})  # consulted only for the lexicon-ambiguous ep_9 side a; answers none


def run_d(tau, records=RECORDS, modes=ALL_MODES, seed=0):
    g = llm()
    params = GenerationParams(seed=seed)
    metric = fake_embedding_metric()
    cfg = build_run_config("D", modes, records, generator=g, lexicon_path=LEXICON_PATH, params=params, similarity=metric, tau=tau)
    return run_stage_attribution(records, "D", load_lexicon(LEXICON_PATH), cfg, generator=g, params=params, similarity=metric, tau=tau)


def run_c(records=RECORDS, modes=ALL_MODES, judge=None):
    g = FakeLLM(judge=judge or (lambda a, b: ("merge", "same")), model_distinction=llm().model_distinction)
    params = GenerationParams(seed=0)
    cfg = build_run_config("C", modes, records, generator=g, lexicon_path=LEXICON_PATH, params=params)
    return run_stage_attribution(records, "C", load_lexicon(LEXICON_PATH), cfg, generator=g, params=params), g


def written(result, tmp):
    write_outputs(Path(tmp) / result.run_config["group_id"], result)
    return load_group(Path(tmp) / result.run_config["group_id"])


def grid_point(points, tau):
    """The sweep point that stands for a run at ``tau``: the largest logged score <= tau (or 'below the smallest')."""
    below = [p for p in points if p["tau"] is not None and p["tau"] <= tau]
    return max(below, key=lambda p: p["tau"]) if below else points[0]


def actual_rates(result, mode):
    return score([json.loads(r.to_json()) for r in result.rows_by_mode[mode]], RECORDS, GOLD)["overall"]


# --- metrics -----------------------------------------------------------------------------------------------------


def row(rid, ua, ub, decision, ea=None, eb=None):
    return {
        "record_id": rid, "decision": decision,
        "entry_a": {"entry_id": ea or f"{ua}_e0", "source_utterance_id": ua},
        "entry_b": {"entry_id": eb or f"{ub}_e0", "source_utterance_id": ub},
    }


def recs(*ids):
    return [rec(i) for i in ids]


class TestMetrics(unittest.TestCase):
    def test_outcomes_follow_the_contract(self):
        records = recs("ep_1", "ep_2", "ep_3", "ep_4", "ep_5")
        rows = [
            row("1", "ep_1_utt_a", "ep_1_utt_b", "merge"),
            row("2", "ep_2_utt_a", "ep_2_utt_b", "no_merge"),
            row("3", "ep_3_utt_a", "ep_3_utt_b", "underdetermined_link"),
            row("4", "ep_4_utt_a", "ep_4_utt_b", "supersede"),
        ]
        out = pair_outcomes(rows, records, no_entry_ids={"ep_5"})
        self.assertEqual(out, {"ep_1": "merged", "ep_2": "compared_not_merged", "ep_3": "compared_not_merged",
                               "ep_4": "merged", "ep_5": "no_entry"})
        self.assertEqual(pair_outcomes(rows, records)["ep_5"], "not_compared")

    def test_chained_merges_connect_the_pair_through_other_entries(self):
        records = recs("ep_1")
        a, b = "ep_1_utt_a", "ep_1_utt_b"
        rows = [  # a_e0 ~ b_e1 and b_e1 ~ b_e0? no: a_e0~b_e0 only through a middle hop a_e0 ~ b_e1 ~ ... b_e0
            row("1", a, b, "merge", ea="a_e0", eb="b_e1"),
            row("2", a, b, "merge", ea="a_e1", eb="b_e1"),
            row("3", a, b, "no_merge", ea="a_e1", eb="b_e0"),
        ]
        out = pair_outcomes(rows, records)
        self.assertEqual(out["ep_1"], "merged")  # a_e0 and a_e1 share b_e1; the pair has a merge path a -> b

    def test_same_utterance_cross_episode_and_unknown_records_are_ignored(self):
        records = recs("ep_1", "ep_2")
        rows = [
            row("1", "ep_1_utt_a", "ep_1_utt_a", "merge", eb="x"),            # same utterance
            row("2", "ep_1_utt_a", "ep_2_utt_b", "merge"),                      # cross-episode
            row("3", "ep_1_utt_a", "unknown_utt", "merge"),                     # unknown utterance
        ]
        self.assertEqual(set(pair_outcomes(rows, records).values()), {"not_compared"})

    def test_rates_use_all_gold_pairs_as_denominators(self):
        gold = gold_from_pairs([GoldPair("ep_1", "different"), GoldPair("ep_2", "different"), GoldPair("ep_3", "same"),
                                GoldPair("ep_4", "same")])
        outcomes = {"ep_1": "merged", "ep_2": "not_compared", "ep_3": "merged", "ep_4": "no_entry"}
        r = rates(outcomes, gold)
        self.assertEqual((r["fcr"], r["mcr"]), (0.5, 0.5))  # dropping pairs cannot lower FCR below merged/all
        self.assertEqual((r["n_different"], r["n_same"], r["false_merged"], r["missed"]), (2, 2, 1, 1))
        self.assertEqual(r["outcomes"]["same"], {"merged": 1, "compared_not_merged": 0, "not_compared": 0, "no_entry": 1})
        self.assertEqual(r["indicative"], {"fcr": True, "mcr": True})  # n < 20

    def test_empty_denominator_is_none_not_zero(self):
        gold = gold_from_pairs([GoldPair("ep_1", "different")])
        r = rates({"ep_1": "compared_not_merged"}, gold)
        self.assertEqual((r["fcr"], r["mcr"], r["n_same"]), (0.0, None, 0))

    def test_indicative_threshold(self):
        gold = gold_from_pairs(GoldPair(f"ep_{i}", "same") for i in range(INDICATIVE_BELOW))
        r = rates({p: "merged" for p in gold}, gold)
        self.assertFalse(r["indicative"]["mcr"])

    def test_unscored_gold_pairs_are_an_error(self):
        with self.assertRaises(GoldError):
            rates({"ep_1": "merged"}, gold_from_pairs([GoldPair("ep_9", "same")]))

    def test_breakdown_by_class_and_language(self):
        out = pair_outcomes([], RECORDS)
        s = score([], RECORDS, GOLD)
        self.assertEqual(set(s["breakdown"]["distinction_class"]), {"kinship", "honorific_register", "name_variant"})
        self.assertEqual(s["breakdown"]["language"]["hi"]["n_same"], 4)
        self.assertEqual(s["breakdown"]["distinction_class"]["kinship"]["n_different"], 3)
        self.assertEqual(len(out), 9)

    def test_gold_file_roundtrip_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "key.jsonl"
            p.write_text('{"eval_pair_id": "a", "relation": "same", "distinction_class": "kinship", "language": "hi"}\n'
                         '{"eval_pair_id": "b", "gold_relation": "different"}\n', encoding="utf-8")
            g = load_gold(p)
            self.assertEqual((g["a"].relation, g["b"].relation, g["a"].distinction_class), ("same", "different", "kinship"))
            p.write_text('{"eval_pair_id": "a", "relation": "maybe"}\n', encoding="utf-8")
            with self.assertRaises(GoldError):
                load_gold(p)
            p.write_text('{"eval_pair_id": "a", "relation": "same"}\n{"eval_pair_id": "a", "relation": "same"}\n', encoding="utf-8")
            with self.assertRaises(GoldError):
                load_gold(p)


# --- the raise-tau sweep -----------------------------------------------------------------------------------------


class TestSweep(unittest.TestCase):
    def test_grid_is_every_distinct_score_plus_one_point_below(self):
        self.assertEqual(tau_grid([0.5, 0.9, 0.5, 0.7]), [None, 0.5, 0.7, 0.9])

    def test_host_decision_is_strictly_greater_than_tau(self):
        gate_off = {"applied": False}
        self.assertEqual(final_decision_at(0.8, 0.8, gate_off), "keep_both")  # sim == tau is not a merge
        self.assertEqual(final_decision_at(0.8, 0.79999, gate_off), "merge")
        self.assertEqual(final_decision_at(0.0, None, gate_off), "merge")
        gate_on = {"applied": True, "distinction_a": {"kinship": "chachi"}, "distinction_b": {"kinship": "mausi"}}
        self.assertEqual(final_decision_at(0.99, 0.5, gate_on), "keep_both")
        under = {"applied": True, "distinction_a": {"kinship": "chachi"}, "distinction_b": {}}
        self.assertEqual(final_decision_at(0.99, 0.5, under), "link_unresolved")
        self.assertEqual(final_decision_at(0.4, 0.5, under), "keep_both")  # apply_gate: no link when the host would not merge
        self.assertIsNone(final_decision_at(0.9, 0.5, {"applied": False, "skipped": "no V4"}))

    def test_sweep_reproduces_real_runs_at_every_tau_in_every_mode(self):
        with tempfile.TemporaryDirectory() as d:
            base = run_d(0.8)
            sweep = sweep_config_d(written(base, d), GOLD)
            self.assertEqual(sweep["n_scores"], 9)
            for tau in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.95, 0.99):
                real = run_d(tau)
                for mode in ALL_MODES:
                    point = grid_point(sweep["modes"][mode]["points"], tau)
                    truth = actual_rates(real, mode)
                    with self.subTest(tau=tau, mode=mode):
                        self.assertEqual((point["fcr"], point["mcr"]), (truth["fcr"], truth["mcr"]))
                        self.assertEqual((point["false_merged"], point["missed"]), (truth["false_merged"], truth["missed"]))

    def test_below_smallest_score_merges_everything_the_gate_allows(self):
        with tempfile.TemporaryDirectory() as d:
            sweep = sweep_config_d(written(run_d(0.8), d), GOLD)
            p0 = sweep["modes"]["off"]["points"][0]
            self.assertIsNone(p0["tau"])
            self.assertEqual((p0["fcr"], p0["mcr"]), (1.0, 0.0))
            last = sweep["modes"]["off"]["points"][-1]
            self.assertEqual((last["tau"], last["fcr"], last["mcr"]), (0.95, 0.0, 1.0))  # tau at the max score merges nothing

    def test_self_check_matches_the_logged_decisions(self):
        with tempfile.TemporaryDirectory() as d:
            sweep = sweep_config_d(written(run_d(0.8), d), GOLD)
            for mode in ALL_MODES:
                self.assertEqual(sweep["modes"][mode]["self_check"]["mismatches"], [])
                self.assertEqual(sweep["modes"][mode]["self_check"]["tau"], 0.8)

    def test_a_tampered_log_fails_the_self_check(self):
        with tempfile.TemporaryDirectory() as d:
            g = written(run_d(0.8), d)
            row0 = next(t for t in g.trace if t["status"] == "ok")
            row0["final"]["off"]["decision"] = "keep_both" if row0["final"]["off"]["decision"] == "merge" else "merge"
            sweep = sweep_config_d(g, GOLD)
            self.assertEqual(sweep["modes"]["off"]["self_check"]["mismatches"], [row0["eval_pair_id"]])

    def test_gate_reduces_fcr_at_every_tau_and_never_raises_it(self):
        with tempfile.TemporaryDirectory() as d:
            sweep = sweep_config_d(written(run_d(0.8), d), GOLD)
            off, lex = sweep["modes"]["off"]["points"], sweep["modes"]["lexicon"]["points"]
            for a, b in zip(off, lex):
                self.assertLessEqual(b["fcr"], a["fcr"])
            self.assertTrue(any(b["fcr"] < a["fcr"] for a, b in zip(off, lex)))

    def test_judge_groups_are_refused(self):
        res, _ = run_c()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SweepError, "no probability/logit threshold"):
                sweep_config_d(written(res, d), GOLD)

    def test_pareto_frontier_and_matched_mcr(self):
        pts = [
            {"tau": None, "fcr": 1.0, "mcr": 0.0}, {"tau": 0.5, "fcr": 0.6, "mcr": 0.1}, {"tau": 0.6, "fcr": 0.7, "mcr": 0.1},
            {"tau": 0.7, "fcr": 0.4, "mcr": 0.3}, {"tau": 0.8, "fcr": 0.4, "mcr": 0.5}, {"tau": 0.9, "fcr": 0.0, "mcr": 1.0},
            {"tau": 0.95, "fcr": None, "mcr": 0.2},
        ]
        fr = pareto_frontier(pts)
        self.assertEqual([(p["mcr"], p["fcr"]) for p in fr], [(0.0, 1.0), (0.1, 0.6), (0.3, 0.4), (1.0, 0.0)])
        self.assertEqual(best_fcr_at_mcr(pts, 0.3)["tau"], 0.7)
        self.assertEqual(best_fcr_at_mcr(pts, 0.05)["fcr"], 1.0)
        self.assertIsNone(best_fcr_at_mcr(pts[1:], 0.05))
        cmp = compare_at_matched_mcr([{"fcr": 0.1, "mcr": 0.3}, {"fcr": 0.9, "mcr": 0.0}], pts)
        self.assertAlmostEqual(cmp[0]["fcr_advantage"], 0.3)  # raise-tau needs 0.4 FCR at MCR <= 0.3; the system has 0.1
        self.assertAlmostEqual(cmp[1]["fcr_advantage"], 0.1)

    def test_dkmem_beats_raise_tau_at_matched_mcr_on_the_synthetic_probe(self):
        with tempfile.TemporaryDirectory() as d:
            sweep = sweep_config_d(written(run_d(0.8), d), GOLD)
            raise_tau = sweep["modes"]["off"]["points"]
            lex_at_run_tau = next(p for p in sweep["modes"]["lexicon"]["points"] if p["tau"] == 0.75)
            cmp = compare_at_matched_mcr([lex_at_run_tau], raise_tau)[0]
            self.assertGreater(cmp["fcr_advantage"], 0)


# --- fairness ------------------------------------------------------------------------------------------------------


class TestFairness(unittest.TestCase):
    def test_same_inputs_model_and_settings_are_comparable(self):
        with tempfile.TemporaryDirectory() as d:
            c, _ = run_c()
            groups = [written(run_d(0.8), d), written(c, d)]
            out = check_comparable(groups)
            self.assertTrue(out["comparable"], out)

    def test_different_seed_inputs_or_decoding_are_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            base = written(run_d(0.8), d)
            other_seed = written(run_d(0.8, seed=1), d)
            fewer = written(run_d(0.8, records=RECORDS[:5]), d)
            keys = {i["key"] for i in check_comparable([base, other_seed])["issues"]}
            self.assertIn("seed", keys)
            keys = {i["key"] for i in check_comparable([base, fewer])["issues"]}
            self.assertTrue({"inputs", "pair_order"} <= keys)
            tweaked = written(run_d(0.8), Path(d) / "tweaked")  # same group id as base: a fresh dir, never an overwrite
            tweaked.run_config["generation_params"]["max_new_tokens"] = 99
            tweaked.run_config["group_id"] = "tweaked"
            self.assertIn("decoding", {i["key"] for i in check_comparable([base, tweaked])["issues"]})

    def test_a_different_embedder_or_lexicon_is_flagged_and_no_llm_groups_are_exempt_from_the_model_keys(self):
        with tempfile.TemporaryDirectory() as d:
            a = written(run_d(0.8), d)
            b = written(run_d(0.9), d)
            b.run_config["embedder"] = {"model_config": {"model_id": "other", "revision": "r"}}
            a.run_config["embedder"] = {"model_config": {"model_id": "BAAI/bge-m3", "revision": "r"}}
            b.run_config["lexicon"] = {"sha256": "deadbeef"}
            keys = {i["key"] for i in check_comparable([a, b])["issues"]}
            self.assertTrue({"embedder", "lexicon"} <= keys)
            no_llm = written(run_d(0.8, modes=["off", "lexicon"]), d)
            no_llm.run_config["backbone"], no_llm.run_config["model_run_info"] = None, None
            no_llm.run_config["group_id"] = "no-llm"
            self.assertNotIn("backbone", {i["key"] for i in check_comparable([a, no_llm])["issues"]})


# --- ablation ------------------------------------------------------------------------------------------------------


class TestAblation(unittest.TestCase):
    def test_report_on_config_d(self):
        with tempfile.TemporaryDirectory() as d:
            report = ablation_report(written(run_d(0.7), d), GOLD, RECORDS)  # at 0.7 ep_9 (0.75) is proposed for merging
        self.assertEqual(report["pairing_violations"], [])
        self.assertEqual(set(report["modes"]), set(ALL_MODES))
        self.assertEqual(report["modes"]["lexicon"]["extra_llm_calls"], 0)
        self.assertEqual(report["modes"]["lexicon+llm"]["extra_llm_calls"], 1)  # only ep_9's ambiguous utterance
        self.assertEqual(report["modes"]["off"]["extra_llm_calls"], 0)
        gate = report["modes"]["lexicon"]["gate"]
        self.assertEqual(sum(gate["compatibility"].values()), gate["pairs_decided"])
        self.assertGreater(gate["vetoed_to_keep_both"], 0)
        # ep_9: side a has two different kinship terms (lexicon-ambiguous); the lexicon picks chachi (compatible with
        # b), the model is asked, answers none -> underdetermined vs b's chachi
        self.assertEqual(report["fallback"]["pairs_whose_compatibility_changes"], 1)
        self.assertEqual(report["fallback"]["changes"][0]["pair"], "ep_9")
        self.assertGreater(report["fallback"]["delta_mcr"], 0)  # the fallback buys safety at the price of a missed same-entity pair

    def test_rates_match_scoring_the_rows_directly(self):
        with tempfile.TemporaryDirectory() as d:
            res = run_d(0.8)
            report = ablation_report(written(res, d), GOLD, RECORDS)
        for mode in ALL_MODES:
            truth = actual_rates(res, mode)
            self.assertEqual((report["modes"][mode]["rates"]["fcr"], report["modes"][mode]["rates"]["mcr"]), (truth["fcr"], truth["mcr"]))

    def test_lexicon_only_never_calls_the_model_for_distinctions(self):
        res, g = run_c(modes=["off", "lexicon"])
        self.assertEqual(g.calls["v4"], 0)  # Config C's gate in lexicon mode needs no V4 call; only the judge ran
        self.assertEqual(g.calls["judge"], len(RECORDS))
        res2, g2 = run_c(modes=["off", "lexicon", "lexicon+llm"])
        self.assertEqual(g2.calls["v4"], 1)  # the model is called only for the one lexicon-ambiguous utterance
        self.assertEqual(g2.calls["judge"], len(RECORDS))
        # same host decisions in both runs: the extra mode changes the gate only
        self.assertEqual([t["host"]["proposed"] for t in res.trace], [t["host"]["proposed"] for t in res2.trace])
        self.assertEqual([t["final"]["lexicon"] for t in res.trace if t["status"] == "ok"],
                         [t["final"]["lexicon"] for t in res2.trace if t["status"] == "ok"])

    def test_pairing_violation_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            g = written(run_d(0.8), d)
            g.rows("lexicon")[0]["similarity_score"] = 0.123
            report = ablation_report(g, GOLD, RECORDS)
            self.assertEqual(len(report["pairing_violations"]), 1)  # the tampered lexicon row disagrees with off


# --- command line ---------------------------------------------------------------------------------------------------


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        self.input = d / "input.jsonl"
        self.input.write_text("\n".join(json.dumps({
            "eval_pair_id": r.eval_pair_id, "language": r.language, "utterance_a": r.utterance_a, "utterance_b": r.utterance_b,
            "utterance_a_id": r.utterance_a_id, "utterance_b_id": r.utterance_b_id,
            "opaque_entity_id_a": r.opaque_entity_id_a, "opaque_entity_id_b": r.opaque_entity_id_b}, ensure_ascii=False)
            for r in RECORDS) + "\n", encoding="utf-8")
        self.gold = d / "key.jsonl"
        self.gold.write_text("\n".join(json.dumps({"eval_pair_id": g.eval_pair_id, "relation": g.relation,
                                                  "distinction_class": g.distinction_class, "language": g.language})
                                       for g in GOLD.values()) + "\n", encoding="utf-8")
        self.d = written(run_d(0.8), d)
        c, _ = run_c()
        self.c = written(c, d)

    def args(self, *a):
        return [*a, "--input", str(self.input), "--gold", str(self.gold)]

    def test_score_sweep_ablation_and_compare(self):
        s = eval_cli.main(["score", "--group", str(self.d.path), str(self.c.path), *self.args()])
        self.assertEqual([g["label"] for g in s["groups"]], ["D", "C"])
        self.assertEqual(set(s["groups"][0]["modes"]), set(ALL_MODES))
        self.assertEqual(s["groups"][0]["modes"]["off"]["strategy"], "embedding-threshold")

        sw = eval_cli.main(["sweep", "--group", str(self.d.path), *self.args()])
        self.assertEqual(sw["sweeps"][0]["run_tau"], 0.8)
        with self.assertRaises(SweepError):
            eval_cli.main(["sweep", "--group", str(self.c.path), *self.args()])

        ab = eval_cli.main(["ablation", "--group", str(self.d.path), *self.args()])
        self.assertEqual(ab["ablations"][0]["pairing_violations"], [])

        out = Path(self.tmp.name) / "compare.json"
        cmp = eval_cli.main(["compare", "--group", str(self.d.path), str(self.c.path), "--d-group", str(self.d.path),
                             "--out", str(out), *self.args()])
        self.assertTrue(cmp["fairness"]["comparable"], cmp["fairness"])
        self.assertEqual(len(cmp["groups"]), 2)
        self.assertIn("vs_raise_tau", cmp["groups"][1]["modes"]["lexicon"])
        self.assertTrue(out.exists())
        self.assertEqual(cmp["raise_tau"]["n_scores"], 9)
        json.loads(out.read_text(encoding="utf-8"))

    def test_compare_requires_the_d_group(self):
        with self.assertRaises(SystemExit):
            eval_cli.main(["compare", "--group", str(self.d.path), *self.args()])


if __name__ == "__main__":
    unittest.main()
