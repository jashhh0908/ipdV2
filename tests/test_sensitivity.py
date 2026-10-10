"""Tests for the register sensitivity replay (dkmem.eval.sensitivity), the key-free diagnostics, and the guards
against overwriting results and reused entry ids.

All pairs and relations below are synthetic test data, not Tier 1 gold.

Run: python -m unittest discover -s tests   (test_pipeline's helpers are imported by name)
"""

import json
import tempfile
import unittest
from pathlib import Path

from test_pipeline import LEXICON_PATH, FakeLLM, rec

from dkmem.backends.llm import GenerationParams
from dkmem.eval import cli as eval_cli
from dkmem.eval.gold import GoldError, GoldPair, gold_from_pairs
from dkmem.eval.groups import load_group
from dkmem.eval.metrics import pair_outcomes
from dkmem.eval.sensitivity import (
    FEATURE_SETS,
    PROVISIONAL_LABEL,
    SensitivityError,
    gate_diagnostics,
    replay_outcomes,
    sensitivity_report,
)
from dkmem.memory.lexicon import load_lexicon
from dkmem.pipeline.runner import build_run_config, run_stage_attribution, write_outputs

ALL_MODES = ["off", "lexicon", "lexicon+llm"]

PAIRS = [
    # id, utterance a, utterance b, synthetic relation, class; the judge proposes "merge" for every pair
    ("ep_1", "meri chachi Pune mein rehti hai", "meri mausi Pune mein rehti hai", "different", "kinship"),  # kinship veto
    ("ep_2", "tu kahan hai", "aap kahan hai", "same", "honorific_register"),                              # register veto
    ("ep_3", "mera bhai Delhi gaya", "mera bhai Delhi gaya tha", "same", "name_variant"),                  # no feature
    ("ep_4", "tum ghar jao", "aap ghar jao", "different", "politeness_relationship"),                      # register veto
    ("ep_5", "meri chachi tu aa ja", "meri mausi aap aa jaiye", "different", "kinship"),                   # both conflict
    ("ep_6", "aap kab aaoge", "woh kab aayega", "same", "honorific_register"),                            # one-sided register
]
RECORDS = [rec(p, a, b) for p, a, b, *_ in PAIRS]
GOLD = gold_from_pairs(GoldPair(p, rel, cls, "hi") for p, _a, _b, rel, cls in PAIRS)


def run_c(records=RECORDS):
    g = FakeLLM(judge=lambda a, b: ("merge", "same"), model_distinction=lambda u: {})
    params = GenerationParams(seed=0)
    cfg = build_run_config("C", ALL_MODES, records, generator=g, lexicon_path=LEXICON_PATH, params=params)
    return run_stage_attribution(records, "C", load_lexicon(LEXICON_PATH), cfg, generator=g, params=params)


class _Group(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.result = run_c()
        self.dir = Path(self.tmp.name) / self.result.run_config["group_id"]
        write_outputs(self.dir, self.result)
        self.group = load_group(self.dir)


class TestReplay(_Group):
    def test_frozen_replay_reproduces_the_logged_outcomes(self):
        for mode in ALL_MODES:
            logged = pair_outcomes(self.group.rows(mode), RECORDS, no_entry_ids=self.group.no_entry_ids())
            self.assertEqual(replay_outcomes(self.group, mode, FEATURE_SETS["frozen"], RECORDS), logged)

    def test_kinship_only_lifts_register_vetoes_and_keeps_kinship_vetoes(self):
        frozen = replay_outcomes(self.group, "lexicon", FEATURE_SETS["frozen"], RECORDS)
        kin = replay_outcomes(self.group, "lexicon", FEATURE_SETS["kinship_only"], RECORDS)
        self.assertEqual(frozen, {"ep_1": "compared_not_merged", "ep_2": "compared_not_merged", "ep_3": "merged",
                                  "ep_4": "compared_not_merged", "ep_5": "compared_not_merged",
                                  "ep_6": "compared_not_merged"})
        self.assertEqual(kin, {**frozen, "ep_2": "merged", "ep_4": "merged", "ep_6": "merged"})

    def test_a_tampered_log_is_refused(self):
        t = next(t for t in self.group.trace if t["eval_pair_id"] == "ep_2")
        t["gate"]["lexicon"]["distinction_b"] = {"register": "tu"}  # now compatible: replay disagrees with the rows
        with self.assertRaises(SensitivityError):
            replay_outcomes(self.group, "lexicon", FEATURE_SETS["kinship_only"], RECORDS)

    def test_non_pipeline_groups_are_refused(self):
        self.group.run_config["pipeline_config"] = None
        with self.assertRaises(SensitivityError):
            replay_outcomes(self.group, "lexicon", FEATURE_SETS["frozen"], RECORDS)


class TestDiagnostics(_Group):
    def test_veto_counts_and_causes_without_the_key(self):
        d = gate_diagnostics(self.group, RECORDS)
        self.assertFalse(d["gold_key_used"])
        self.assertNotIn("gate", d["modes"]["off"])
        self.assertEqual(d["modes"]["off"]["outcomes"]["merged"], 6)
        g = d["modes"]["lexicon"]["gate"]
        self.assertEqual((g["pairs_gated"], g["host_proposed_merge"], g["vetoed"]), (6, 6, 5))
        self.assertEqual(g["vetoed_by_cause"], {"kinship": 1, "kinship+register": 1, "register": 3})
        self.assertEqual(g["vetoed_to"], {"keep_both": 4, "link_unresolved": 1})
        self.assertEqual(g["vetoed_if_kinship_only"], 2)
        self.assertEqual(g["vetoed_by_language"], {"hi": 5})

    def test_output_carries_no_labels_and_no_pair_ids(self):
        text = json.dumps(gate_diagnostics(self.group, RECORDS))
        for leak in ('"same"', '"different"', "relation", "distinction_class"):
            self.assertNotIn(leak, text)
        self.assertNotRegex(text, r"ep_\d")  # no pair ids


class TestSensitivityReport(_Group):
    def test_rates_and_paired_tests(self):
        r = sensitivity_report(self.group, GOLD, RECORDS)
        self.assertEqual(r["label"], PROVISIONAL_LABEL)
        self.assertEqual(r["off"]["all"]["fcr"], 1.0)
        frozen, kin = r["modes"]["lexicon"]["frozen"], r["modes"]["lexicon"]["kinship_only"]
        self.assertEqual((frozen["all"]["rates"]["fcr"], frozen["all"]["rates"]["mcr"]), (0.0, 2 / 3))
        self.assertEqual((kin["all"]["rates"]["fcr"], kin["all"]["rates"]["mcr"]), (1 / 3, 0.0))
        self.assertEqual(kin["all"]["paired_vs_frozen"]["mcr"]["only_a"], 2)  # misses only the frozen gate makes
        self.assertEqual(kin["all"]["paired_vs_frozen"]["fcr"]["only_b"], 1)  # ep_4: the register veto was right
        self.assertNotIn("paired_vs_frozen", frozen["all"])
        excl = frozen["excluding_honorific_register"]["rates"]
        self.assertEqual((excl["n_same"], excl["n_different"]), (1, 3))
        self.assertEqual(frozen["by_class"]["honorific_register"]["rates"]["mcr"], 1.0)
        self.assertIsNone(frozen["by_class"]["honorific_register"]["rates"]["fcr"])  # n = 0 -> n/a

    def test_report_is_aggregate_only(self):
        text = json.dumps(sensitivity_report(self.group, GOLD, RECORDS))
        self.assertNotRegex(text, r"ep_\d")  # no pair ids


class TestNoOverwrite(_Group):
    def test_write_outputs_refuses_an_existing_group(self):
        before = (self.dir / "run_config.json").read_text(encoding="utf-8")
        with self.assertRaises(FileExistsError):
            write_outputs(self.dir, run_c())
        self.assertEqual((self.dir / "run_config.json").read_text(encoding="utf-8"), before)

    def test_reused_entry_id_across_utterances_is_an_error(self):
        rows = [
            {"record_id": "1", "decision": "no_merge", "entry_a": {"entry_id": "e0", "source_utterance_id": "ep_1_utt_a"},
             "entry_b": {"entry_id": "x", "source_utterance_id": "ep_1_utt_b"}},
            {"record_id": "2", "decision": "no_merge", "entry_a": {"entry_id": "e0", "source_utterance_id": "ep_2_utt_a"},
             "entry_b": {"entry_id": "y", "source_utterance_id": "ep_2_utt_b"}},
        ]
        with self.assertRaises(GoldError):
            pair_outcomes(rows, [rec("ep_1"), rec("ep_2")])


class TestCli(_Group):
    def setUp(self):
        super().setUp()
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

    def test_diagnostics_runs_without_the_key(self):
        out = Path(self.tmp.name) / "diag.json"
        rep = eval_cli.main(["diagnostics", "--group", str(self.dir), "--input", str(self.input), "--out", str(out)])
        self.assertEqual(rep["diagnostics"][0]["modes"]["lexicon"]["gate"]["vetoed"], 5)
        self.assertTrue(out.exists())

    def test_sensitivity_needs_the_key(self):
        with self.assertRaises(SystemExit):
            eval_cli.main(["sensitivity", "--group", str(self.dir), "--input", str(self.input)])
        rep = eval_cli.main(["sensitivity", "--group", str(self.dir), "--input", str(self.input), "--gold", str(self.gold)])
        self.assertEqual(rep["sensitivity"][0]["label"], PROVISIONAL_LABEL)

    def test_existing_out_file_is_never_overwritten(self):
        out = Path(self.tmp.name) / "diag.json"
        out.write_text("prior", encoding="utf-8")
        with self.assertRaises(SystemExit):
            eval_cli.main(["diagnostics", "--group", str(self.dir), "--input", str(self.input), "--out", str(out)])
        self.assertEqual(out.read_text(encoding="utf-8"), "prior")


if __name__ == "__main__":
    unittest.main()
