"""Regression: the memory store reproduces the frozen A-D 3B run (``results/abcd_3b_s0``).

Each frozen Tier 1 episode is replayed through ``dkmem.store`` as two writes into a fresh
store (utterance a, then utterance b) -- the shape of a one-episode conversation -- and the
result is compared with what the frozen pipeline wrote:

* every ``pairwise_eval.jsonl`` row of every group and DK-Mem mode (decision, compatibility,
  similarity score, threshold, ids, distinctions, gold pass-through, run id, strategy), exactly;
* every trace row's gate block, host proposal and judge record, final decision and loss-point
  diagnosis;
* the rows dropped in some modes only (``lexicon+llm`` without a V4 extraction): the store skips
  that write, so the row is absent, as in the frozen run;
* the counts in ``summary.json``.

What is replayed rather than recomputed: the model-dependent inputs -- the stored texts (gloss /
native fact / utterance), the V4 model's own distinction, the judge's raw answers and, in
Config D, the bge-m3 cosines -- all read from the frozen trace. What the store path computes
itself: the lexicon distinctions, retrieval, the host proposal from the judge answer or from
``sim > tau``, the gate, the single-target action, the diagnosis and the export. In A-C the
similarity is the real ``DEFAULT_SIMILARITY`` (difflib), unchanged.

The frozen results are gitignored, so this test is skipped where ``results/abcd_3b_s0`` is absent.

Run: python -m unittest discover -s tests
"""

import json
import unittest
from pathlib import Path

from dkmem.backends.llm import GenerationParams
from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.schema import Extraction
from dkmem.memory.similarity import DEFAULT_SIMILARITY, SimilarityMetric
from dkmem.pipeline.runner import mode_slug, strategy_for
from dkmem.pipeline.trace import sha256_file, sha256_file_lf
from dkmem.store.consolidate import (
    ConsolidationConfig,
    JudgeHost,
    ThresholdHost,
    consolidate,
    entry_distinction,
)
from dkmem.store.diagnosis import pair_diagnoser
from dkmem.store.entry import StoredEntry
from dkmem.store.export import pairwise_records
from dkmem.store.store import MemoryStore
from dkmem.tier1.io import TIER1_INPUT_SHA256, load_tier1_input

REPO = Path(__file__).parent.parent
FROZEN = REPO / "results" / "abcd_3b_s0" / "runs"
LEXICON_PATH = REPO / "dkmem" / "memory" / "distinction_features.json"
INPUT_PATH = REPO / "dkmem" / "data" / "tier1" / "eval_input" / "teamA_tier1_v2.jsonl"
K = 5  # irrelevant to Tier 1 (a store of one entry) but exercised: retrieval must return that entry


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


class ReplayJudgeGenerator:
    """Answers a judge prompt with the raw output the frozen run recorded for the same two texts."""

    backbone = "replay"

    def __init__(self, trace):
        self.answers = {}
        for t in trace:
            j = (t.get("host") or {}).get("judge")
            if j is not None:
                key = f"Memory A: {t['stored']['a']['text']}\nMemory B: {t['stored']['b']['text']}"
                self.answers.setdefault(key, set()).add(j["raw_output"])
        self.asked = []

    def generate(self, prompts, params=None):
        out = []
        for p in prompts:
            key = p[-1]["content"]
            raws = self.answers[key]  # KeyError: the store asked something the frozen run never did
            assert len(raws) == 1, f"frozen run gave different answers to the same texts: {key!r}"
            self.asked.append(key)
            out.append(next(iter(raws)))
        return out


def replay_metric(trace):
    scores = {}
    for t in trace:
        if t.get("similarity"):
            s = t["similarity"]
            for k in ((s["text_a"], s["text_b"]), (s["text_b"], s["text_a"])):
                scores[k] = s["score"]
    return SimilarityMetric("bge_m3_dense_cosine_v1", lambda a, b: scores[(a, b)])


def v4_from_trace(block, utterance, side, seed):
    """The V4 model's own distinction as an ``Extraction`` (None when the V4 call failed in the frozen run)."""
    if block.get("model_distinction") is None:
        return None
    return Extraction(
        pair_id="replay", side=side, raw_output="", gloss="", distinction=block["model_distinction"],
        surface=utterance, lang_profile={}, backbone="replay", prompt_id=block["v4_prompt_id"], seed=seed,
    )


def replay_group(group_dir, lexicon, gold):
    rc = json.loads((group_dir / "run_config.json").read_text(encoding="utf-8"))
    trace = read_jsonl(group_dir / "trace.jsonl")
    config_id, tau, seed = rc["pipeline_config"]["config_id"], rc["tau"], rc["seed"]
    modes = rc["dkmem_modes"]
    judge_gen = None
    if rc["tau_decides_merges"]:
        metric, host = replay_metric(trace), ThresholdHost(tau)
    else:
        judge_gen = ReplayJudgeGenerator(trace)
        metric, host = DEFAULT_SIMILARITY, JudgeHost(judge_gen, GenerationParams(seed=seed))
    diagnoser = pair_diagnoser(lexicon, config_id)

    stores = {m: [] for m in modes}
    by_episode = {}  # (mode, eval_pair_id) -> (store, write_a, write_b)
    for t in trace:
        if t["status"] not in ("ok", "judge_failed"):
            continue  # extraction failed / unsupported language: no entry was stored on either side
        inp, lang = t["input"], t["language"]
        for mode in modes:
            store = MemoryStore(t["eval_pair_id"], meta={"config": config_id, "mode": mode, "tau": tau})
            writes = []
            for side, utt_key, id_key in (("a", "utterance_a", "utterance_a_id"), ("b", "utterance_b", "utterance_b_id")):
                dist = entry_distinction(
                    mode, utterance=inp[utt_key], language=lang, lexicon=lexicon,
                    v4=v4_from_trace(t["extraction"][side], inp[utt_key], side, seed),
                )
                e = StoredEntry(
                    entry_id=t["stored"][side]["entry_id"], conv_id=t["eval_pair_id"],
                    source_utterance_id=inp[id_key], language=lang,
                    stored_text=t["stored"][side]["text"], surface=inp[utt_key], distinction=dist,
                )
                writes.append(consolidate(store, e, host, metric, ConsolidationConfig(mode, K), diagnoser=diagnoser))
            stores[mode].append(store)
            by_episode[(mode, t["eval_pair_id"])] = (store, *writes)

    rows = {}
    for mode in modes:
        events = [ev for s in stores[mode] for ev in s.events]
        rows[mode] = pairwise_records(
            events, mode=mode, run_id=f"{rc['group_id']}-{mode_slug(mode)}",
            strategy=strategy_for(config_id, mode), threshold=tau, gold=gold,
        )
    return {"rc": rc, "trace": trace, "stores": stores, "by_episode": by_episode, "rows": rows, "judge": judge_gen}


@unittest.skipUnless(FROZEN.is_dir(), "frozen results/abcd_3b_s0 not present (gitignored)")
class TestStoreReproducesFrozenAD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexicon = load_lexicon(LEXICON_PATH)
        records = load_tier1_input(INPUT_PATH)  # hash-checked
        cls.gold = {}
        for r in records:
            cls.gold[r.utterance_a_id] = r.opaque_entity_id_a
            cls.gold[r.utterance_b_id] = r.opaque_entity_id_b
        cls.groups = sorted(p for p in FROZEN.iterdir() if p.is_dir())
        cls.replays = {g.name: replay_group(g, cls.lexicon, cls.gold) for g in cls.groups}

    def test_the_frozen_run_is_the_one_this_environment_can_replay(self):
        self.assertEqual(len(self.groups), 6)
        self.assertEqual(sha256_file(INPUT_PATH.resolve()), TIER1_INPUT_SHA256)
        for g in self.groups:
            rc = self.replays[g.name]["rc"]
            self.assertEqual(rc["lexicon"]["sha256"], sha256_file_lf(LEXICON_PATH), f"{g.name}: lexicon changed since the run")
            self.assertEqual(rc["input"]["file_sha256"], TIER1_INPUT_SHA256)
            self.assertEqual(rc["input"]["n_records"], 173)

    def test_pairwise_rows_are_identical_in_every_group_and_mode(self):
        total = 0
        for g in self.groups:
            rep = self.replays[g.name]
            for mode in rep["rc"]["dkmem_modes"]:
                with self.subTest(group=g.name, mode=mode):
                    frozen = read_jsonl(g / mode_slug(mode) / "pairwise_eval.jsonl")
                    ours = [json.loads(r.to_json()) for r in rep["rows"][mode]]
                    self.assertEqual(len(ours), len(frozen))
                    self.assertEqual(sorted(ours, key=lambda r: r["record_id"]), sorted(frozen, key=lambda r: r["record_id"]))
                    total += len(frozen)
        self.assertGreater(total, 3000)  # 6 groups x 3 modes x ~172 rows

    def test_decisions_compatibility_and_similarity_match_the_trace(self):
        for g in self.groups:
            rep = self.replays[g.name]
            for t in rep["trace"]:
                if t["status"] not in ("ok", "judge_failed"):
                    continue
                for mode in rep["rc"]["dkmem_modes"]:
                    with self.subTest(group=g.name, mode=mode, pair=t["eval_pair_id"]):
                        _store, wa, wb = rep["by_episode"][(mode, t["eval_pair_id"])]
                        if t["status"] == "judge_failed":
                            comp = wb.event["candidates"][0]
                            self.assertEqual((comp["status"], comp["final"]), ("host_failed", None))
                            self.assertTrue(wb.event["had_failures"])
                            continue
                        gate = t["gate"][mode]
                        if gate.get("skipped"):  # lexicon+llm without a V4 extraction on a side
                            self.assertIn("skipped", {wa.action, wb.action})
                            self.assertIsNone(t["final"][mode])
                            continue
                        comp = wb.event["candidates"][0]
                        self.assertEqual(comp["candidate_id"], t["stored"]["a"]["entry_id"])
                        self.assertEqual(comp["score"], t["similarity"]["score"])
                        self.assertEqual(comp["host"]["proposed"], t["host"]["proposed"])
                        if t["host"]["judge"] is not None:
                            self.assertEqual(comp["host"]["detail"], t["host"]["judge"])
                        else:
                            self.assertEqual(comp["host"]["detail"], {"tau": t["host"]["tau"]})
                        self.assertEqual(comp["gate"], gate)
                        self.assertEqual(comp["final"], t["final"][mode]["decision"])
                        self.assertEqual(comp["diagnosis"], t["diagnosis"])

    def test_pairs_dropped_in_only_some_modes_are_skipped_writes(self):
        dropped = 0
        for g in self.groups:
            rep = self.replays[g.name]
            for t in rep["trace"]:
                if t["status"] != "ok":
                    continue
                for mode in rep["rc"]["dkmem_modes"]:
                    if t["final"][mode] is None:
                        dropped += 1
                        store, wa, wb = rep["by_episode"][(mode, t["eval_pair_id"])]
                        self.assertEqual(mode, "lexicon+llm")
                        self.assertTrue({wa.event["event"], wb.event["event"]} & {"write_skipped"})
                        self.assertEqual(store.stats()["writes_skipped"], sum(w.action == "skipped" for w in (wa, wb)))
        self.assertGreater(dropped, 0)  # the Korean pair whose V4 call failed, in B, C and D

    def test_summary_counts_and_action_shapes(self):
        for g in self.groups:
            rep = self.replays[g.name]
            summary = json.loads((g / "summary.json").read_text(encoding="utf-8"))
            for mode in rep["rc"]["dkmem_modes"]:
                with self.subTest(group=g.name, mode=mode):
                    self.assertEqual(len(rep["rows"][mode]), summary["rows_by_mode"][mode])
                    from collections import Counter

                    ours = dict(sorted(Counter(r.decision for r in rep["rows"][mode]).items()))
                    self.assertEqual(ours, summary["final_decisions"][mode])
                    for store in rep["stores"][mode]:
                        self.assertEqual(store.integrity_errors(), [])
                        if mode != "off":
                            self.assertEqual(store.gate_conflicts(), [], store.conv_id)
                        acts = [ev["action"] for ev in store.events if ev["event"] in ("write", "write_skipped")]
                        self.assertEqual(acts[0], "add" if acts[0] != "skipped" else "skipped")
                        self.assertLessEqual(len(store.active()), 2)

    def test_the_judge_was_only_asked_what_the_frozen_run_asked(self):
        for g in self.groups:
            rep = self.replays[g.name]
            if rep["judge"] is not None:  # A-C: a KeyError would already have failed setUpClass
                self.assertTrue(rep["judge"].asked)
                self.assertTrue(set(rep["judge"].asked) <= set(rep["judge"].answers))


if __name__ == "__main__":
    unittest.main()
