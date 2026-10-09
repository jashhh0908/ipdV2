"""FCR and MCR exactly as ``dkmem/tier1_eval_contract.md`` Sec 4-6 define them, for any run's ``pairwise_eval.jsonl``.

Outcome of a pair (Sec 4): clusters are built by union-find over ``entry_id`` using every record whose decision is
``merge`` or ``supersede``, after dropping same-utterance, cross-episode and unknown-utterance records. The pair is

* ``merged``               some entry of utterance a and some entry of utterance b share a cluster;
* ``compared_not_merged``  a cross-utterance record exists but no merge path (``no_merge`` / ``underdetermined_link``);
* ``not_compared``         no record links the two utterances;
* ``no_entry``             an utterance produced zero entries (told apart from ``not_compared`` only when the caller
                           passes ``no_entry_ids``, e.g. from the trace; both count as not merged).

``FCR = #{gold-different pairs that are merged} / #{gold-different pairs}`` and
``MCR = #{gold-same pairs that are not merged} / #{gold-same pairs}``; the denominators are **all** gold pairs,
whether or not the system compared them, so a system cannot lower either rate by dropping pairs. A rate whose
denominator is 0 is ``None`` (reported ``n/a``). Cells with fewer than ``INDICATIVE_BELOW`` pairs of a relation are
flagged. ``link_unresolved`` / ``underdetermined_link`` is *not merged*.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dkmem.eval.gold import GoldError, GoldPair
from dkmem.tier1.io import Tier1Record

__all__ = [
    "OUTCOMES",
    "UNIFYING_DECISIONS",
    "INDICATIVE_BELOW",
    "read_jsonl",
    "pair_outcomes",
    "rates",
    "breakdown",
    "score",
]

OUTCOMES = ("merged", "compared_not_merged", "not_compared", "no_entry")
UNIFYING_DECISIONS = ("merge", "supersede")
INDICATIVE_BELOW = 20


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def pair_outcomes(
    rows: Iterable[Mapping[str, Any]],
    records: Sequence[Tier1Record],
    *,
    no_entry_ids: Iterable[str] = (),
) -> dict[str, str]:
    """The outcome of every record's pair (see the module docstring); ``rows`` are ``pairwise_eval`` dicts."""
    utt_pair: dict[str, str] = {}
    for r in records:
        utt_pair[r.utterance_a_id] = r.eval_pair_id
        utt_pair[r.utterance_b_id] = r.eval_pair_id
    uf = _UnionFind()
    sources: dict[str, set[str]] = {}  # pair -> entry ids, with their source utterance remembered below
    entry_src: dict[str, str] = {}
    compared: set[str] = set()
    for row in rows:
        ea, eb = row["entry_a"], row["entry_b"]
        ua, ub = ea.get("source_utterance_id"), eb.get("source_utterance_id")
        if ua not in utt_pair or ub not in utt_pair or ua == ub or utt_pair[ua] != utt_pair[ub]:
            continue  # unknown utterance, same-utterance and cross-episode records are ignored
        pair = utt_pair[ua]
        compared.add(pair)
        for e, u in ((ea, ua), (eb, ub)):
            entry_src[e["entry_id"]] = u
            sources.setdefault(pair, set()).add(e["entry_id"])
        if row["decision"] in UNIFYING_DECISIONS:
            uf.union(ea["entry_id"], eb["entry_id"])
    no_entry = set(no_entry_ids)
    out: dict[str, str] = {}
    for r in records:
        entries = sources.get(r.eval_pair_id, set())
        roots_a = {uf.find(e) for e in entries if entry_src[e] == r.utterance_a_id}
        roots_b = {uf.find(e) for e in entries if entry_src[e] == r.utterance_b_id}
        if roots_a & roots_b:
            out[r.eval_pair_id] = "merged"
        elif r.eval_pair_id in compared:
            out[r.eval_pair_id] = "compared_not_merged"
        elif r.eval_pair_id in no_entry:
            out[r.eval_pair_id] = "no_entry"
        else:
            out[r.eval_pair_id] = "not_compared"
    return out


def _check_universe(outcomes: Mapping[str, str], gold: Mapping[str, GoldPair]) -> None:
    unknown = sorted(set(gold) - set(outcomes))
    if unknown:
        raise GoldError(f"{len(unknown)} gold pair(s) are not among the scored pairs, e.g. {unknown[:3]}")


def rates(outcomes: Mapping[str, str], gold: Mapping[str, GoldPair], pair_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """FCR / MCR over ``pair_ids`` (default: every gold pair); the denominators are all of those pairs."""
    _check_universe(outcomes, gold)
    ids = list(gold) if pair_ids is None else list(pair_ids)
    by_rel: dict[str, Counter] = {"different": Counter(), "same": Counter()}
    for p in ids:
        by_rel[gold[p].relation][outcomes[p]] += 1
    n_diff, n_same = sum(by_rel["different"].values()), sum(by_rel["same"].values())
    false_merged = by_rel["different"]["merged"]
    missed = n_same - by_rel["same"]["merged"]
    return {
        "fcr": false_merged / n_diff if n_diff else None,
        "mcr": missed / n_same if n_same else None,
        "n_different": n_diff, "n_same": n_same, "false_merged": false_merged, "missed": missed,
        "indicative": {"fcr": 0 < n_diff < INDICATIVE_BELOW, "mcr": 0 < n_same < INDICATIVE_BELOW},
        "outcomes": {rel: {o: c[o] for o in OUTCOMES} for rel, c in by_rel.items()},
    }


def breakdown(
    outcomes: Mapping[str, str], gold: Mapping[str, GoldPair], by: Sequence[str] = ("distinction_class", "language")
) -> dict[str, Any]:
    """FCR/MCR per value of each field in ``by`` (``distinction_class`` / ``language``) and per class x language cell."""
    _check_universe(outcomes, gold)
    out: dict[str, Any] = {}
    for field in by:
        groups: dict[str, list[str]] = {}
        for p, g in gold.items():
            groups.setdefault(str(getattr(g, field)), []).append(p)
        out[field] = {k: rates(outcomes, gold, ids) for k, ids in sorted(groups.items())}
    if set(by) >= {"distinction_class", "language"}:
        cells: dict[str, list[str]] = {}
        for p, g in gold.items():
            cells.setdefault(f"{g.distinction_class} x {g.language}", []).append(p)
        out["distinction_class x language"] = {k: rates(outcomes, gold, ids) for k, ids in sorted(cells.items())}
    return out


def score(
    rows: Iterable[Mapping[str, Any]],
    records: Sequence[Tier1Record],
    gold: Mapping[str, GoldPair],
    *,
    no_entry_ids: Iterable[str] = (),
    by: Sequence[str] = ("distinction_class", "language"),
) -> dict[str, Any]:
    """Overall rates plus the per-class / per-language / cell breakdown for one ``pairwise_eval`` file."""
    outcomes = pair_outcomes(rows, records, no_entry_ids=no_entry_ids)
    return {"overall": rates(outcomes, gold), "breakdown": breakdown(outcomes, gold, by), "outcomes": outcomes}
