
"""Reliability additions to the offline evaluation (dkmem.eval): confidence intervals, the paired gate-on / gate-off
comparison, the key-vs-input alignment check and the contract's gold passthrough check.

Why they exist: Tier 1 v2 has cells of 4 to 13 pairs (name_variant-different 4, honorific_register 7 / 13, German 6),
a headline claim that is a *paired* comparison (the gate against the same host without it), and a contract that rejects
a run whose entries' ``gold_entity_id`` contradicts the key. Synthetic data only. The one test that reads Team B's real
key (never committed; ``teamB_answer_key.jsonl`` at the repository root) is skipped when the file is absent and asserts
only structure and the counts the contract already publishes.

Run: python -m unittest discover -s tests
"""

import copy
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from test_eval import GOLD, RECORDS, run_c, run_d, written
from test_pipeline import rec

from dkmem.eval import cli as eval_cli
from dkmem.eval.ablation import ablation_report
from dkmem.eval.fairness import check_comparable, manifest_problems, pin_warnings
from dkmem.eval.gold import GoldError, GoldPair, check_alignment, gold_from_pairs, load_gold
from dkmem.eval.groups import GroupData, load_group
from dkmem.eval.metrics import (
    check_passthrough,
    mcnemar_exact,
    pair_outcomes,
    paired_difference,
    rates,
    score,
    wilson_interval,
)
from dkmem.tier1.io import load_tier1_input

REPO = Path(__file__).parent.parent
KEY = REPO / "teamB_answer_key.jsonl"
INPUT = REPO / "dkmem" / "data" / "tier1" / "eval_input" / "teamA_tier1_v2.jsonl"


class TestIntervals(unittest.TestCase):
    def test_wilson_known_values(self):
        lo, hi = wilson_interval(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        lo, hi = wilson_interval(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        lo, hi = wilson_interval(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.7225, places=3)

    def test_empty_cell_has_no_interval_and_small_cells_stay_wide(self):
        self.assertIsNone(wilson_interval(0, 0))
        lo, hi = wilson_interval(2, 4)  # the 4 name_variant-different pairs
        self.assertGreater(hi - lo, 0.5)
        for n in (4, 6, 13, 32, 98):
            for k in range(n + 1):
                lo, hi = wilson_interval(k, n)
                self.assertTrue(0.0 <= lo <= k / n <= hi <= 1.0, (k, n))

    def test_rates_carry_intervals_and_n_zero_is_none(self):
        gold = gold_from_pairs([GoldPair("ep_1", "different"), GoldPair("ep_2", "different")])
        r = rates({"ep_1": "merged", "ep_2": "compared_not_merged"}, gold)
        self.assertEqual(r["fcr"], 0.5)
        lo, hi = r["fcr_ci95"]
        self.assertTrue(lo < 0.5 < hi)
        self.assertEqual((r["mcr"], r["mcr_ci95"]), (None, None))

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)
        self.assertEqual(mcnemar_exact(1, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(5, 0), 0.0625)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 0.03125)
        self.assertEqual(mcnemar_exact(2, 7), mcnemar_exact(7, 2))
        self.assertAlmostEqual(mcnemar_exact(1, 9), 2 * 11 / 1024)


class TestPairedDifference(unittest.TestCase):
    def setUp(self):
        # 8 different pairs, 4 same pairs
        self.gold = gold_from_pairs(
            [GoldPair(f"d{i}", "different", "kinship", "hi") for i in range(8)]
            + [GoldPair(f"s{i}", "same", "honorific_register", "hi") for i in range(4)]
        )
        merged, kept = "merged", "compared_not_merged"
        # reference: merges d0..d5 (6 false merges) and s0,s1 (2 of 4 same merged -> 2 missed)
        self.a = {**{f"d{i}": merged if i < 6 else kept for i in range(8)}, **{f"s{i}": merged if i < 2 else kept for i in range(4)}}
        # system: removes d0..d4 (5 false merges avoided), keeps d5; also blocks s1 (one more miss)
        self.b = {**{f"d{i}": merged if i == 5 else kept for i in range(8)}, **{f"s{i}": merged if i == 0 else kept for i in range(4)}}

    def test_counts_rates_and_p_value(self):
        p = paired_difference(self.a, self.b, self.gold)
        f, m = p["fcr"], p["mcr"]
        self.assertEqual((f["only_a"], f["only_b"], f["both"], f["neither"], f["n"]), (5, 0, 1, 2, 8))
        self.assertEqual((f["rate_a"], f["rate_b"], f["delta"]), (6 / 8, 1 / 8, -5 / 8))
        self.assertAlmostEqual(f["p_mcnemar"], 0.0625)
        self.assertEqual((m["only_a"], m["only_b"], m["both"], m["neither"]), (0, 1, 2, 1))  # s1 newly missed; s2, s3 always
        self.assertEqual((m["rate_a"], m["rate_b"], m["delta"]), (2 / 4, 3 / 4, 1 / 4))
        self.assertEqual(m["p_mcnemar"], 1.0)

    def test_agrees_with_rates(self):
        p = paired_difference(self.a, self.b, self.gold)
        ra, rb = rates(self.a, self.gold), rates(self.b, self.gold)
        self.assertAlmostEqual(p["fcr"]["delta"], rb["fcr"] - ra["fcr"])
        self.assertAlmostEqual(p["mcr"]["delta"], rb["mcr"] - ra["mcr"])
        self.assertEqual(p["fcr"]["rate_a"], ra["fcr"])

    def test_identical_systems_have_no_difference(self):
        p = paired_difference(self.a, dict(self.a), self.gold)
        self.assertEqual((p["fcr"]["delta"], p["fcr"]["p_mcnemar"], p["mcr"]["delta"]), (0.0, 1.0, 0.0))

    def test_subset_and_empty_relation(self):
        p = paired_difference(self.a, self.b, self.gold, pair_ids=[f"d{i}" for i in range(8)])
        self.assertEqual(p["mcr"]["n"], 0)
        self.assertIsNone(p["mcr"]["delta"])
        self.assertIsNone(p["mcr"]["p_mcnemar"])

    def test_missing_pair_is_an_error(self):
        with self.assertRaises(GoldError):
            paired_difference({k: v for k, v in self.a.items() if k != "d0"}, self.b, self.gold)

    def test_an_error_means_miss_for_same_pairs_and_false_merge_for_different(self):
        gold = gold_from_pairs([GoldPair("x", "same"), GoldPair("y", "different")])
        p = paired_difference({"x": "merged", "y": "merged"}, {"x": "no_entry", "y": "no_entry"}, gold)
        self.assertEqual((p["mcr"]["only_b"], p["fcr"]["only_a"]), (1, 1))


class TestAblationReportsThePairedEffect(unittest.TestCase):
    def test_paired_vs_off_matches_the_modes_own_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            group = written(run_d(0.8), Path(tmp))
            report = ablation_report(group, GOLD, RECORDS)
        off = report["modes"]["off"]
        self.assertNotIn("paired_vs_off", off)
        for mode in ("lexicon", "lexicon+llm"):
            m = report["modes"][mode]
            p = m["paired_vs_off"]
            self.assertAlmostEqual(p["fcr"]["delta"], m["rates"]["fcr"] - off["rates"]["fcr"])
            self.assertAlmostEqual(p["mcr"]["delta"], m["rates"]["mcr"] - off["rates"]["mcr"])
            self.assertEqual(p["fcr"]["only_b"], 0)  # the gate only vetoes: it never creates a false merge
        self.assertGreater(report["modes"]["lexicon"]["paired_vs_off"]["fcr"]["only_a"], 0)


class TestRunIntegrity(unittest.TestCase):
    def rows(self):
        c, _ = run_c(modes=["off"])
        return [json.loads(r.to_json()) for r in c.rows_by_mode["off"]]

    def test_honest_rows_pass(self):
        rows = self.rows()
        check_passthrough(rows, RECORDS)
        self.assertEqual(score(rows, RECORDS, GOLD)["overall"]["n_different"], 5)

    def test_a_wrong_gold_entity_id_rejects_the_run(self):
        rows = self.rows()
        rows[0]["entry_a"]["gold_entity_id"] = "ent_somebody_else"
        with self.assertRaises(GoldError):
            check_passthrough(rows, RECORDS)
        with self.assertRaises(GoldError):
            score(rows, RECORDS, GOLD)

    def test_a_missing_gold_entity_id_rejects_the_run(self):
        rows = self.rows()
        del rows[1]["entry_b"]["gold_entity_id"]
        with self.assertRaises(GoldError):
            check_passthrough(rows, RECORDS)

    def test_entries_of_unknown_utterances_are_ignored_as_in_scoring(self):
        rows = self.rows()
        rows[0]["entry_b"]["source_utterance_id"] = "not_in_the_input"
        rows[0]["entry_b"]["gold_entity_id"] = "whatever"
        check_passthrough(rows, RECORDS)
        self.assertEqual(pair_outcomes(rows, RECORDS)[RECORDS[0].eval_pair_id], "not_compared")


class TestKeyAlignment(unittest.TestCase):
    def key_row(self, r, **over):
        d = {"eval_pair_id": r.eval_pair_id, "relation": "same", "distinction_class": "kinship", "language": r.language,
             "utterance_a_id": r.utterance_a_id, "utterance_b_id": r.utterance_b_id,
             "opaque_entity_id_a": r.opaque_entity_id_a, "opaque_entity_id_b": r.opaque_entity_id_b}
        d.update(over)
        return d

    def load(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "key.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            return load_gold(p)

    def test_aligned_key_passes_and_keeps_the_id_fields(self):
        gold = self.load([self.key_row(r) for r in RECORDS])
        check_alignment(gold, RECORDS)
        g = gold[RECORDS[0].eval_pair_id]
        self.assertEqual((g.utterance_a_id, g.opaque_entity_id_b), (RECORDS[0].utterance_a_id, RECORDS[0].opaque_entity_id_b))

    def test_each_disagreement_is_detected(self):
        r = RECORDS[0]
        for over in ({"language": "tr"}, {"utterance_a_id": "x"}, {"utterance_b_id": "x"},
                     {"opaque_entity_id_a": "ent_x"}, {"opaque_entity_id_b": "ent_x"}):
            with self.subTest(over=over):
                with self.assertRaises(GoldError):
                    check_alignment(self.load([self.key_row(r, **over)]), RECORDS)

    def test_key_pair_without_a_record_is_an_error_but_unkeyed_records_are_fine(self):
        extra = rec("ep_not_in_input")
        with self.assertRaises(GoldError):
            check_alignment(self.load([self.key_row(extra)]), RECORDS)
        check_alignment(self.load([self.key_row(RECORDS[0])]), RECORDS)  # the other records are simply not scored

    def test_minimal_key_without_optional_fields_still_loads_and_passes(self):
        gold = self.load([{"eval_pair_id": RECORDS[0].eval_pair_id, "relation": "same"}])
        check_alignment(gold, RECORDS)

    def test_cli_rejects_a_misaligned_key_and_accepts_an_aligned_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            inp = d / "input.jsonl"
            inp.write_text("\n".join(json.dumps({
                "eval_pair_id": r.eval_pair_id, "language": r.language, "utterance_a": r.utterance_a,
                "utterance_b": r.utterance_b, "utterance_a_id": r.utterance_a_id, "utterance_b_id": r.utterance_b_id,
                "opaque_entity_id_a": r.opaque_entity_id_a, "opaque_entity_id_b": r.opaque_entity_id_b}) for r in RECORDS) + "\n",
                encoding="utf-8")
            group = written(run_d(0.8), d)

            def key(path, **over):
                rows = [self.key_row(r, relation=GOLD[r.eval_pair_id].relation, **(over if r is RECORDS[2] else {})) for r in RECORDS]
                path.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
                return path

            good, bad = key(d / "good.jsonl"), key(d / "bad.jsonl", opaque_entity_id_a="ent_regenerated")
            args = lambda k: ["score", "--group", str(group.path), "--input", str(inp), "--gold", str(k)]  # noqa: E731
            out = eval_cli.main(args(good))
            self.assertIn("lexicon", out["groups"][0]["modes"])
            with self.assertRaises(GoldError):
                eval_cli.main(args(bad))


class TestReproducibilityChecks(unittest.TestCase):
    """fairness.check_comparable: pair coverage, lexicon+llm policy, prompt hashes, manifests, revision pins."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        self.d = written(run_d(0.8), d)
        c, _ = run_c()
        self.c = written(c, d)

    def clone(self, group, **changes):
        cfg = copy.deepcopy(group.run_config)
        for path, value in changes.items():
            node = cfg
            *parents, last = path.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            if value is _DELETE:
                node.pop(last, None)
            else:
                node[last] = value
        return GroupData(group.path, cfg, copy.deepcopy(group.trace))

    def issues(self, *groups):
        return {i["key"] for i in check_comparable(list(groups))["issues"]}

    def test_two_groups_with_the_same_id_are_not_merged(self):
        # a rerun has the original's fingerprint and so its group id: a difference between them must still show
        rerun = self.clone(self.d, seed=1)
        self.assertEqual(rerun.group_id, self.d.group_id)
        report = check_comparable([self.d, rerun])
        self.assertIn("seed", {i["key"] for i in report["issues"]})
        groups = [g for i in report["issues"] if i["key"] == "seed" for v in i["values"] for g in v["groups"]]
        self.assertEqual(sorted(groups), sorted([f"{self.d.group_id}#0", f"{self.d.group_id}#1"]))

    def test_untouched_groups_have_no_issues(self):
        report = check_comparable([self.d, self.c])
        self.assertEqual(report["issues"], [])
        self.assertTrue(report["comparable"])

    def test_trace_row_order_is_not_a_difference_but_coverage_is(self):
        # the A-D harness writes failed and skipped episodes before the decided ones: same episodes, other sequence
        shuffled = self.clone(self.d)
        shuffled.trace.reverse()
        self.assertEqual(self.issues(self.d, shuffled), set())
        missing = self.clone(self.d)
        missing.trace.pop(0)
        self.assertIn("pair_order", self.issues(self.d, missing))

    def test_legacy_and_current_lexicon_llm_policy_are_not_comparable(self):
        self.assertIn("lexicon_llm_policy", self.d.run_config)
        legacy = self.clone(self.d, lexicon_llm_policy=_DELETE)
        self.assertIn("lexicon_llm_policy", self.issues(self.d, legacy))
        self.assertNotIn("lexicon_llm_policy", self.issues(self.d, self.clone(self.d)))

    def test_an_edited_frozen_prompt_is_an_issue_but_two_prompts_are_not(self):
        prompts = self.c.run_config["prompts"]
        role = next(r for r, p in prompts.items() if p["prompt_id"] == "merge_judge_v1")
        edited = self.clone(self.c, **{f"prompts.{role}.sha256": "0" * 64})
        report = check_comparable([self.c, edited])
        self.assertEqual([i["prompt_id"] for i in report["issues"] if i["key"] == "prompt_hash"], ["merge_judge_v1"])
        informed = self.clone(self.c, **{f"prompts.{role}": {"prompt_id": "merge_judge_informed_v1", "sha256": "1" * 64}})
        self.assertNotIn("prompt_hash", self.issues(self.c, informed))

    def test_manifest_contradicting_the_run_config_is_an_issue(self):
        self.assertEqual(manifest_problems(self.d), [])
        shutil.copytree(self.d.path, Path(self.tmp.name) / "copy")
        copy_group = load_group(Path(self.tmp.name) / "copy")
        mpath = copy_group.path / "lexicon" / "run_manifest.json"
        m = json.loads(mpath.read_text(encoding="utf-8"))
        m["seed"], m["run_id"] = 99, "someone-else"
        mpath.write_text(json.dumps(m), encoding="utf-8")
        problems = manifest_problems(copy_group)
        self.assertTrue(any("seed" in p for p in problems) and any("run_id" in p for p in problems), problems)
        self.assertTrue(any("pairwise_eval.jsonl differ" in p for p in problems), problems)
        self.assertIn("manifests", self.issues(copy_group))

    def test_missing_output_file_is_reported(self):
        shutil.copytree(self.d.path, Path(self.tmp.name) / "copy2")
        g = load_group(Path(self.tmp.name) / "copy2")
        (g.path / "off" / "run_manifest.json").unlink()
        self.assertTrue(any("missing output file" in p for p in manifest_problems(g)))

    def test_pin_warnings(self):
        d_unrecorded = self.clone(self.c, git={"commit": None, "dirty": True})
        text = " | ".join(pin_warnings(d_unrecorded))
        self.assertIn("weights revision not recorded", text)  # the fake model records no resolved revision
        self.assertIn("dirty git checkout", text)
        self.assertIn("no git commit recorded", text)
        not_pinned = self.clone(self.c, **{"model_run_info.resolved_revision": "abc"})
        self.assertIn("not pinned on the command line (resolved abc)", " ".join(pin_warnings(not_pinned)))
        mismatch = self.clone(self.c, **{"model_run_info.model_config.revision": "x", "model_run_info.resolved_revision": "abc"})
        self.assertIn("requested revision x but resolved abc", " ".join(pin_warnings(mismatch)))
        fully = self.clone(self.c, **{"model_run_info.model_config.revision": "abc", "model_run_info.resolved_revision": "abc",
                                      "git": {"commit": "c0ffee", "dirty": False}})
        self.assertEqual(pin_warnings(fully), [])

    def test_warnings_and_notes_do_not_change_comparable(self):
        other_tree = self.clone(self.c, **{"code.tree_sha256": "f" * 64})
        report = check_comparable([self.c, other_tree])
        self.assertTrue(report["comparable"])
        self.assertTrue(any("different code trees" in n for n in report["notes"]))
        self.assertTrue(report["warnings"])
        unsanctioned = self.clone(self.c, **{"input.sanctioned": False})
        self.assertTrue(any("sanctioned" in n for n in check_comparable([self.c, unsanctioned])["notes"]))


_DELETE = object()


@unittest.skipUnless(KEY.is_file(), "Team B's key (teamB_answer_key.jsonl at the repository root) is not present")
class TestTeamBKeyAgainstTheSanctionedInput(unittest.TestCase):
    """Structure and published counts only; nothing from the key is printed or copied anywhere."""

    @classmethod
    def setUpClass(cls):
        cls.gold = load_gold(KEY)
        cls.records = load_tier1_input(INPUT)

    def test_the_key_matches_the_input_exactly(self):
        check_alignment(self.gold, self.records)
        self.assertEqual(set(self.gold), {r.eval_pair_id for r in self.records})
        for g in self.gold.values():
            self.assertTrue(all(getattr(g, f) for f in ("language", "distinction_class", "utterance_a_id", "opaque_entity_id_b")))

    def test_counts_are_the_contracts(self):
        self.assertEqual(Counter(g.relation for g in self.gold.values()), {"different": 98, "same": 75})
        self.assertEqual(Counter(g.language for g in self.gold.values()), {"hi": 145, "ko": 12, "tr": 10, "de": 6})
        per_class = Counter((g.distinction_class, g.relation) for g in self.gold.values())
        expected = {("evidentiality", "same"): 34, ("kinship", "different"): 32, ("name_variant", "same"): 20,
                    ("name_variant", "different"): 4, ("classifier_measure", "different"): 24,
                    ("politeness_relationship", "same"): 14, ("politeness_relationship", "different"): 10,
                    ("spatial_temporal_deixis", "different"): 22, ("honorific_register", "same"): 7,
                    ("honorific_register", "different"): 6}
        self.assertEqual(dict(per_class), expected)

    def test_an_empty_run_scores_every_pair_as_not_compared(self):
        s = score([], self.records, self.gold)
        self.assertEqual(set(s["outcomes"].values()), {"not_compared"})
        self.assertEqual((s["overall"]["n_different"], s["overall"]["n_same"]), (98, 75))


if __name__ == "__main__":
    unittest.main()
