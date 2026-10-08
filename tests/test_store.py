"""Tests for the in-memory DK-Mem memory store (dkmem.store): store operations, retrieval,
the consolidation flow (host-first gate, single-target merge, unresolved links, failures),
hosts, and the pairwise / log export. No model or GPU.

The regression against the frozen A-D 3B run is in test_store_regression.py.

Run: python -m unittest discover -s tests
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import jsonschema

from dkmem.memory.similarity import SimilarityError, SimilarityMetric
from dkmem.store.consolidate import (
    ConsolidationConfig,
    HostProposal,
    HostQuery,
    JudgeHost,
    ThresholdHost,
    consolidate,
    entry_distinction,
)
from dkmem.store.entry import Link, StoredEntry
from dkmem.store.export import pairwise_records, write_store_log
from dkmem.store.retrieval import retrieve
from dkmem.store.store import MemoryStore, StoreError

REPO = Path(__file__).parent.parent
PAIRWISE_SCHEMA = json.loads((REPO / "dkmem" / "pairwise_eval.schema.json").read_text(encoding="utf-8"))

CHACHI, MAUSI = {"kinship": "chachi"}, {"kinship": "mausi"}


def entry(eid, text=None, dist=None, utt=None, conv="c1", lang="hi"):
    return StoredEntry(
        entry_id=eid, conv_id=conv, source_utterance_id=utt or f"{eid}_utt", language=lang,
        stored_text=text or f"text of {eid}", surface=f"surface of {eid}",
        distinction=dist if dist is not None else {},
    )


def TableMetric(table=None):
    """Symmetric score table keyed by the two texts; unlisted pairs score 0."""
    scores = {frozenset(k): v for k, v in (table or {}).items()}
    return SimilarityMetric("table", lambda a, b: scores.get(frozenset((a, b)), 0.0))


class ScriptedHost:
    """Answers by candidate text; records the queries it was asked."""

    mechanism = "scripted"

    def __init__(self, answers=None, default="merge"):
        self.answers, self.default, self.queries = answers or {}, default, []

    def propose(self, queries):
        self.queries.append(list(queries))
        out = []
        for q in queries:
            d = self.answers.get(q.candidate_text, self.default)
            out.append(HostProposal(d, {"scripted": True}))
        return out


def write(store, e, host=None, metric=None, mode="lexicon", k=5, **kw):
    return consolidate(store, e, host or ScriptedHost(), metric or TableMetric(), ConsolidationConfig(mode, k), **kw)


def seeded_store(*entries, conv="c1"):
    store = MemoryStore(conv)
    for i, e in enumerate(entries):
        e.seq = i
        store.add(e)
    return store


class TestStore(unittest.TestCase):
    def test_add_get_active_and_members(self):
        s = seeded_store(entry("e1"), entry("e2"))
        self.assertEqual([e.entry_id for e in s.active()], ["e1", "e2"])
        self.assertEqual([m["entry_id"] for m in s.get("e1").members], ["e1"])
        with self.assertRaises(StoreError):
            s.get("nope")

    def test_duplicate_id_and_wrong_conversation_rejected(self):
        s = seeded_store(entry("e1"))
        with self.assertRaises(StoreError):
            s.add(entry("e1"))
        with self.assertRaises(StoreError):
            s.add(entry("e9", conv="other"))
        with self.assertRaises(StoreError):
            MemoryStore("  ")

    def test_absorb_replaces_text_keeps_history_and_tombstones(self):
        s = seeded_store(entry("e1", "old text", CHACHI))
        new = entry("e2", "new text", CHACHI)
        new.seq = 1
        s.absorb("e1", new)
        t = s.get("e1")
        self.assertEqual((t.stored_text, t.surface), ("new text", "surface of e2"))
        self.assertEqual(t.history[0]["replaced_text"], "old text")
        self.assertEqual(t.history[0]["by_entry_id"], "e2")
        self.assertEqual([m["entry_id"] for m in t.members], ["e1", "e2"])
        self.assertEqual(t.distinction, CHACHI)
        self.assertEqual((s.get("e2").status, s.get("e2").absorbed_by), ("absorbed", "e1"))
        self.assertEqual([e.entry_id for e in s.active()], ["e1"])
        self.assertEqual(s.integrity_errors(), [])

    def test_cannot_absorb_into_a_tombstone_or_link_one(self):
        s = seeded_store(entry("e1"), entry("e3"))
        s.absorb("e1", entry("e2"))
        with self.assertRaises(StoreError):
            s.absorb("e2", entry("e4"))
        with self.assertRaises(StoreError):
            s.link("e2", "e3", seq=0, reason="x")
        with self.assertRaises(StoreError):
            s.absorb("e1", entry("e5"), decision="keep_both")

    def test_links_are_undirected_and_deduplicated(self):
        s = seeded_store(entry("e1"), entry("e2"))
        self.assertTrue(s.link("e2", "e1", seq=1, reason="r"))
        self.assertFalse(s.link("e1", "e2", seq=2, reason="again"))
        self.assertEqual(len(s.links), 1)
        self.assertEqual((s.links[0].a, s.links[0].b), ("e1", "e2"))
        self.assertEqual(len(s.links_of("e1")), 1)
        with self.assertRaises(ValueError):
            Link("e1", "e1", 0, "x")

    def test_integrity_errors_detect_tampering(self):
        s = seeded_store(entry("e1"), entry("e2"))
        s.link("e1", "e2", seq=0, reason="r")
        s.absorb("e1", entry("e3"))
        self.assertEqual(s.integrity_errors(), [])
        s.get("e3").absorbed_by = "e2"
        self.assertTrue(s.integrity_errors())

    def test_snapshot_stats_and_event_log_are_json_serializable(self):
        s = seeded_store(entry("e1"), entry("e2"))
        s.link("e1", "e2", seq=1, reason="r")
        s.absorb("e1", entry("e3"))
        json.dumps(s.snapshot(), ensure_ascii=False)
        json.dumps(s.events, ensure_ascii=False)
        self.assertEqual(
            s.stats(),
            {"entries_total": 3, "entries_active": 2, "entries_absorbed": 1, "merged_clusters": 1,
             "unresolved_links": 1, "writes": 0, "writes_skipped": 0},
        )
        self.assertEqual([ev["n"] for ev in s.events], list(range(len(s.events))))


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.store = seeded_store(entry("a", "ta"), entry("b", "tb"), entry("c", "tc"))

    def test_ranks_by_score_best_first_and_limits_k(self):
        m = TableMetric({("ta", "q"): 0.2, ("tb", "q"): 0.9, ("tc", "q"): 0.5})
        got = retrieve(self.store, "q", m, 2)
        self.assertEqual([(c.entry_id, c.rank, c.score) for c in got], [("b", 0, 0.9), ("c", 1, 0.5)])

    def test_ties_keep_insertion_order(self):
        m = TableMetric({("ta", "q"): 0.5, ("tb", "q"): 0.5, ("tc", "q"): 0.5})
        self.assertEqual([c.entry_id for c in retrieve(self.store, "q", m, 3)], ["a", "b", "c"])

    def test_excludes_same_utterance_including_merged_members(self):
        self.store.absorb("a", entry("d", "td", utt="utt_d"))
        m = TableMetric()
        self.assertEqual([c.entry_id for c in retrieve(self.store, "q", m, 5, exclude_utterance_ids={"a_utt"})], ["b", "c"])
        self.assertEqual([c.entry_id for c in retrieve(self.store, "q", m, 5, exclude_utterance_ids={"utt_d"})], ["b", "c"])

    def test_absorbed_entries_are_not_retrievable(self):
        self.store.absorb("a", entry("d", "td"))
        self.assertNotIn("d", [c.entry_id for c in retrieve(self.store, "q", TableMetric(), 5)])

    def test_k_validated_and_out_of_range_score_is_an_error(self):
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                retrieve(self.store, "q", TableMetric(), bad)
        with self.assertRaises(SimilarityError):
            retrieve(self.store, "q", SimilarityMetric("bad", lambda a, b: 2.0), 1)

    def test_empty_store_returns_nothing(self):
        self.assertEqual(retrieve(MemoryStore("c1"), "q", TableMetric(), 3), [])


class TestConsolidation(unittest.TestCase):
    def test_first_write_adds_and_never_asks_the_host(self):
        s, host = MemoryStore("c1"), ScriptedHost()
        r = write(s, entry("e1"), host)
        self.assertEqual((r.action, r.target_id), ("add", None))
        self.assertEqual(host.queries, [])
        self.assertEqual(len(s.active()), 1)

    def test_compatible_merge_newer_text_wins_old_text_in_history(self):
        s = MemoryStore("c1")
        write(s, entry("e1", "Pune fact", CHACHI))
        r = write(s, entry("e2", "Delhi fact", CHACHI), metric=TableMetric({("Pune fact", "Delhi fact"): 0.9}))
        self.assertEqual((r.action, r.target_id, r.event["decision"]), ("merge", "e1", "merge"))
        self.assertEqual([e.entry_id for e in s.active()], ["e1"])
        t = s.get("e1")
        self.assertEqual(t.stored_text, "Delhi fact")
        self.assertEqual(t.history[0]["replaced_text"], "Pune fact")
        self.assertEqual(r.event["overwritten"]["replaced_text"], "Pune fact")
        self.assertEqual(r.event["target_members_before"], ["e1_utt"])
        self.assertEqual(s.integrity_errors(), [])

    def test_incompatible_merge_is_vetoed_to_keep_both_without_a_link(self):
        s = MemoryStore("c1")
        write(s, entry("e1", "a", CHACHI))
        r = write(s, entry("e2", "b", MAUSI))
        comp = r.event["candidates"][0]
        self.assertEqual((comp["host"]["proposed"], comp["final"]), ("merge", "keep_both"))
        self.assertEqual((comp["gate"]["compatibility"], comp["gate"]["vetoed"]), ("incompatible", True))
        self.assertEqual((r.action, len(s.active()), len(s.links)), ("add", 2, 0))

    def test_underdetermined_merge_becomes_an_unresolved_link(self):
        s = MemoryStore("c1")
        write(s, entry("e1", "a", {}))
        r = write(s, entry("e2", "b", CHACHI))
        comp = r.event["candidates"][0]
        self.assertEqual((comp["final"], comp["gate"]["compatibility"]), ("link_unresolved", "underdetermined"))
        self.assertEqual((r.action, len(s.active())), ("add", 2))
        self.assertEqual([(l.a, l.b) for l in s.links], [("e1", "e2")])
        self.assertIn("kinship", s.links[0].reason)
        self.assertEqual(r.event["links"][0]["new"], True)
        self.assertEqual(s.stats()["unresolved_links"], 1)

    def test_no_link_when_the_host_would_not_have_merged(self):
        s = MemoryStore("c1")
        write(s, entry("e1", "a", {}))
        r = write(s, entry("e2", "b", CHACHI), ScriptedHost(default="keep_both"))
        comp = r.event["candidates"][0]
        self.assertEqual((comp["final"], comp["gate"]["compatibility"], comp["gate"]["vetoed"]),
                         ("keep_both", "underdetermined", False))
        self.assertEqual(len(s.links), 0)

    def test_mode_off_does_not_gate_and_merges_across_a_distinction(self):
        s = MemoryStore("c1")
        write(s, entry("e1", "a", CHACHI), mode="off")
        r = write(s, entry("e2", "b", MAUSI), mode="off")
        comp = r.event["candidates"][0]
        self.assertEqual(r.action, "merge")
        self.assertEqual(comp["gate"], {"applied": False, "proposed": "merge", "decision": "merge"})

    def test_gate_conflicts_empty_with_gate_nonempty_without(self):
        def run(mode):
            s = MemoryStore("c1")
            write(s, entry("e1", "a", CHACHI), mode=mode)
            write(s, entry("e2", "b", MAUSI), mode=mode)
            return s

        self.assertEqual(run("lexicon").gate_conflicts(), [])
        self.assertEqual(len(run("off").gate_conflicts()), 1)

    def test_gated_write_without_a_distinction_is_skipped_not_stored_and_not_judged(self):
        s, host = MemoryStore("c1"), ScriptedHost()
        write(s, entry("e1", "a", CHACHI), host, mode="lexicon+llm")
        host.queries.clear()
        bad = entry("e2", "b", None)
        bad.distinction = None
        r = write(s, bad, host, mode="lexicon+llm")
        self.assertEqual((r.action, r.event["event"]), ("skipped", "write_skipped"))
        self.assertIn("not written", r.event["reason"])
        self.assertEqual(host.queries, [])
        self.assertNotIn("e2", s.entries)
        self.assertEqual((len(s.active()), s.stats()["writes_skipped"], s.stats()["writes"]), (1, 1, 2))

    def test_mode_off_does_not_need_a_distinction(self):
        s = MemoryStore("c1")
        e = entry("e1")
        e.distinction = None
        self.assertEqual(write(s, e, mode="off").action, "add")

    def test_host_failure_is_logged_not_decided(self):
        s = MemoryStore("c1")
        write(s, entry("e1", "a", CHACHI))
        r = write(s, entry("e2", "b", CHACHI), ScriptedHost(default=None))
        comp = r.event["candidates"][0]
        self.assertEqual((comp["status"], comp["final"], comp["gate"]), ("host_failed", None, None))
        self.assertTrue(r.event["had_failures"])
        self.assertEqual((r.action, len(s.active()), len(s.links)), ("add", 2, 0))

    def test_single_target_is_the_highest_ranked_survivor_and_links_follow_the_survivor(self):
        # new entry marks kinship=chachi. A: unmarked (underdetermined), B: mausi (incompatible),
        # C and D: chachi (compatible). The host would merge with all four.
        s = seeded_store(
            entry("A", "tA", {}), entry("B", "tB", MAUSI), entry("C", "tC", CHACHI), entry("D", "tD", CHACHI)
        )
        m = TableMetric({("tA", "N"): 0.9, ("tB", "N"): 0.8, ("tC", "N"): 0.7, ("tD", "N"): 0.6})
        r = write(s, entry("N", "N", CHACHI), metric=m)
        finals = {c["candidate_id"]: c["final"] for c in r.event["candidates"]}
        self.assertEqual(finals, {"A": "link_unresolved", "B": "keep_both", "C": "merge", "D": "merge"})
        self.assertEqual((r.action, r.target_id), ("merge", "C"))
        self.assertEqual(s.get("D").status, "active")  # not merged into a second cluster
        self.assertEqual([m_["entry_id"] for m_ in s.get("D").members], ["D"])
        self.assertEqual([(l.a, l.b) for l in s.links], [("A", "C")])
        self.assertEqual([e.entry_id for e in s.active()], ["A", "B", "C", "D"])
        self.assertEqual(s.integrity_errors(), [])

    def test_k_limits_which_candidates_are_compared(self):
        s = seeded_store(entry("A", "tA", MAUSI), entry("B", "tB", MAUSI), entry("C", "tC", CHACHI))
        m = TableMetric({("tA", "N"): 0.9, ("tB", "N"): 0.8, ("tC", "N"): 0.7})
        r = write(s, entry("N", "N", CHACHI), metric=m, k=2)
        self.assertEqual([c["candidate_id"] for c in r.event["candidates"]], ["A", "B"])
        self.assertEqual(r.action, "add")  # C would have been compatible but is outside the top 2

    def test_same_utterance_entries_are_not_compared(self):
        s = MemoryStore("c1")
        write(s, entry("e1", "a", CHACHI, utt="u1"))
        r = write(s, entry("e2", "b", CHACHI, utt="u1"))
        self.assertEqual((r.action, r.event["candidates"]), ("add", []))

    def test_diagnoser_hook_runs_per_decided_comparison_only(self):
        seen = []

        def diag(cand, new, proposed):
            seen.append((cand.entry_id, new.entry_id, proposed))
            return {"x": 1}

        s = seeded_store(entry("A", "tA", CHACHI), entry("B", "tB", CHACHI))
        host = ScriptedHost({"tA": None, "tB": "keep_both"})
        r = write(s, entry("N", "N", CHACHI), host, TableMetric({("tA", "N"): 0.9, ("tB", "N"): 0.8}), diagnoser=diag)
        self.assertEqual(seen, [("B", "N", "keep_both")])
        self.assertEqual([c["diagnosis"] for c in r.event["candidates"]], [None, {"x": 1}])

    def test_host_must_answer_every_query(self):
        class Short:
            mechanism = "short"

            def propose(self, queries):
                return []

        s = seeded_store(entry("A", "tA"))
        with self.assertRaises(ValueError):
            write(s, entry("N", "N"), Short(), TableMetric({("tA", "N"): 0.5}))

    def test_config_validation(self):
        for bad in (dict(mode="gate"), dict(mode="off", k=0), dict(mode="off", k=True)):
            with self.assertRaises(ValueError):
                ConsolidationConfig(**bad)

    def test_flat_control_never_merging_host_keeps_every_entry(self):
        s, host = MemoryStore("c1"), ScriptedHost(default="keep_both")
        for i in range(6):
            write(s, entry(f"e{i}", f"t{i}", CHACHI), host, TableMetric(), mode="off")
        self.assertEqual(len(s.active()), 6)

    def test_event_log_replays_to_the_same_active_set_and_links(self):
        rng = random.Random(7)
        s = MemoryStore("c1")
        dists = [{}, CHACHI, MAUSI, {"register": "formal"}, {"register": "informal"}, {"kinship": "chachi", "register": "formal"}]
        texts = [f"t{i}" for i in range(12)]
        table = {(a, b): rng.random() for i, a in enumerate(texts) for b in texts[i + 1:]}
        host = ScriptedHost()
        for i in range(30):
            host.default = rng.choice(["merge", "merge", "keep_both", None])
            write(s, entry(f"e{i}", rng.choice(texts), rng.choice(dists)), host, TableMetric(table), k=rng.choice([1, 3, 5]))
        self.assertEqual(s.integrity_errors(), [])
        self.assertEqual(s.gate_conflicts(), [])  # the gate keeps every cluster's discriminative marks consistent
        active, links = set(), set()
        for ev in s.events:
            if ev["event"] == "add":
                active.add(ev["entry_id"])
            elif ev["event"] == "link":
                links.add((ev["a"], ev["b"]))
        self.assertEqual(active, {e.entry_id for e in s.active()})
        self.assertEqual(links, {(l.a, l.b) for l in s.links})
        writes = [ev for ev in s.events if ev["event"] == "write"]
        self.assertEqual(len(writes), 30)
        self.assertEqual(s.stats()["entries_total"], 30)


class TestHosts(unittest.TestCase):
    def test_threshold_host_is_strictly_greater_than_tau(self):
        h = ThresholdHost(0.85)
        qs = [HostQuery("c", "x", "y", s) for s in (0.85, 0.8500001, 0.2)]
        self.assertEqual([p.decision for p in h.propose(qs)], ["keep_both", "merge", "keep_both"])
        self.assertEqual(h.propose(qs)[0].detail, {"tau": 0.85})
        for bad in (-0.1, 1.1, True, None, "0.5"):
            with self.assertRaises(ValueError):
                ThresholdHost(bad)

    def test_judge_host_shows_the_stored_entry_as_memory_a_and_the_new_one_as_memory_b(self):
        prompts = []

        class Gen:
            backbone = "fake"

            def generate(self, ps, params=None):
                prompts.extend(ps)
                return ['{"decision": "merge", "reason": "same"}', "not json"]

        out = JudgeHost(Gen()).propose([HostQuery("c1", "old text", "new text", 0.5), HostQuery("c2", "o2", "n2", 0.4)])
        self.assertEqual(prompts[0][-1]["content"], "Memory A: old text\nMemory B: new text")
        self.assertEqual(out[0].decision, "merge")
        self.assertEqual(out[0].detail["prompt_id"], "merge_judge_v1")
        self.assertIsNone(out[1].decision)
        self.assertIn("not a single JSON object", out[1].detail["error"])
        self.assertEqual(JudgeHost(Gen()).propose([]), [])

    def test_host_proposal_validates_its_decision(self):
        with self.assertRaises(ValueError):
            HostProposal("supersede")


class TestEntryDistinction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dkmem.memory.lexicon import load_lexicon

        cls.lex = load_lexicon(REPO / "dkmem" / "memory" / "distinction_features.json")

    def test_modes(self):
        from dkmem.memory.schema import Extraction

        utt = "meri chachi Pune mein rehti hai"
        kw = dict(utterance=utt, language="hi", lexicon=self.lex)
        self.assertEqual(entry_distinction("off", **kw), {})
        self.assertEqual(entry_distinction("lexicon", **kw), {"kinship": "chachi"})
        self.assertIsNone(entry_distinction("lexicon+llm", **kw))  # no V4 -> unavailable, never a lexicon fallback
        v4 = Extraction(pair_id="p", side="a", raw_output="", gloss="g", distinction={"register": "formal"},
                        surface=utt, lang_profile={}, backbone="b", prompt_id="mem0_extraction_v4", seed=0)
        self.assertEqual(entry_distinction("lexicon+llm", v4=v4, **kw), {"kinship": "chachi", "register": "formal"})
        with self.assertRaises(ValueError):
            entry_distinction("gate", **kw)


class TestExport(unittest.TestCase):
    GOLD = {"e1_utt": "ent_1", "e2_utt": "ent_2", "e3_utt": "ent_3"}

    def run_store(self, mode="lexicon", host=None):
        s = MemoryStore("c1")
        write(s, entry("e1", "a", CHACHI), host, mode=mode)
        write(s, entry("e2", "b", MAUSI), host, mode=mode)
        write(s, entry("e3", "c", CHACHI), host, mode=mode)
        return s

    def rows(self, s, mode="lexicon", threshold=None):
        return pairwise_records(s.events, mode=mode, run_id="run-x", strategy="dk-mem-lexicon", threshold=threshold, gold=self.GOLD)

    def test_rows_follow_candidate_then_new_orientation_and_validate_against_the_schema(self):
        rows = self.rows(self.run_store())
        self.assertEqual(len(rows), 3)  # e2 vs e1; e3 vs e1 and e3 vs e2 (k=5)
        for r in rows:
            jsonschema.validate(r.to_dict(), PAIRWISE_SCHEMA)
        first = rows[0]
        self.assertEqual((first.record_id, first.decision, first.compatibility), ("e1::e2", "no_merge", "incompatible"))
        self.assertEqual((first.entry_a.entry_id, first.entry_b.entry_id), ("e1", "e2"))
        self.assertEqual((first.entry_a.gold_entity_id, first.entry_b.gold_entity_id), ("ent_1", "ent_2"))
        self.assertEqual(first.entry_a.distinction.to_dict()["extraction_method"], "lexicon")
        self.assertIsNone(first.predicted_entity_id)
        merged = [r for r in rows if r.decision == "merge"]
        self.assertEqual([(r.record_id, r.predicted_entity_id) for r in merged], [("e1::e3", "e1")])

    def test_mode_off_rows_have_null_compatibility_and_distinction(self):
        s = self.run_store("off")
        rows = pairwise_records(s.events, mode="off", run_id="r", strategy="mem0", threshold=None, gold=self.GOLD)
        self.assertTrue(rows)
        for r in rows:
            self.assertIsNone(r.compatibility)
            self.assertIsNone(r.entry_a.distinction)
            jsonschema.validate(r.to_dict(), PAIRWISE_SCHEMA)

    def test_threshold_is_copied_and_validated(self):
        rows = self.rows(self.run_store(), threshold=0.85)
        self.assertEqual({r.threshold for r in rows}, {0.85})

    def test_failed_comparisons_and_skipped_writes_make_no_rows(self):
        s = MemoryStore("c1")
        write(s, entry("e1", "a", CHACHI))
        write(s, entry("e2", "b", CHACHI), ScriptedHost(default=None))
        skipped = entry("e3", "c", None)
        skipped.distinction = None
        write(s, skipped)
        self.assertEqual(self.rows(s), [])

    def test_missing_gold_is_an_error_not_a_blank(self):
        s = self.run_store()
        with self.assertRaises(KeyError):
            pairwise_records(s.events, mode="lexicon", run_id="r", strategy="dk-mem-lexicon", threshold=None, gold={})

    def test_store_log_files_roundtrip(self):
        s = self.run_store()
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_store_log(tmp, s)
            events = [json.loads(l) for l in Path(paths["events"]).read_text(encoding="utf-8").splitlines()]
            final = json.loads(Path(paths["final"]).read_text(encoding="utf-8"))
        self.assertEqual(events, json.loads(json.dumps(s.events)))
        self.assertEqual(final["stats"]["entries_active"], 2)
        self.assertEqual(final["conv_id"], "c1")


if __name__ == "__main__":
    unittest.main()
