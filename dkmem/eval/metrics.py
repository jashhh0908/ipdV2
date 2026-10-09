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
import math
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
    "wilson_interval",
    "mcnemar_exact",
    "rates",
    "paired_difference",
    "check_passthrough",
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


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval (95% by default) for ``k`` of ``n``; ``None`` when ``n`` is 0.

    Deterministic and well behaved at the small cells of Tier 1 (4 to 98 pairs), where a normal approximation would
    leave ``[0, 1]``. It treats pairs as independent draws from the probe distribution: it says how far a rate could
    move on other pairs of the same kind, not whether two systems differ (use ``paired_difference`` for that).
    """
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # the exact interval always contains p; clamp so rounding at k = 0 or k = n cannot leave it out
    return [min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half))]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts (1.0 when there are none)."""
    d = only_a + only_b
    if d == 0:
        return 1.0
    tail = sum(math.comb(d, i) for i in range(min(only_a, only_b) + 1)) / 2**d
    return min(1.0, 2 * tail)


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
        "fcr_ci95": wilson_interval(false_merged, n_diff), "mcr_ci95": wilson_interval(missed, n_same),
        "n_different": n_diff, "n_same": n_same, "false_merged": false_merged, "missed": missed,
        "indicative": {"fcr": 0 < n_diff < INDICATIVE_BELOW, "mcr": 0 < n_same < INDICATIVE_BELOW},
        "outcomes": {rel: {o: c[o] for o in OUTCOMES} for rel, c in by_rel.items()},
    }


def paired_difference(
    outcomes_a: Mapping[str, str], outcomes_b: Mapping[str, str], gold: Mapping[str, GoldPair],
    pair_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """System ``b`` against reference ``a`` (e.g. gate ``lexicon`` against ``off``) on the *same* gold pairs.

    An *error* is a false merge for a gold-``different`` pair and a missed merge for a gold-``same`` pair. Per rate
    (``fcr`` over the different pairs, ``mcr`` over the same pairs): ``only_a`` / ``only_b`` count the pairs on which
    just that system errs, ``both`` / ``neither`` the rest; ``delta`` is the rate of ``b`` minus the rate of ``a``
    (negative: ``b`` errs less); ``p_mcnemar`` is the exact two-sided McNemar p-value on the discordant pairs. This is
    the test for "the gate lowers FCR": it uses the pairing, so it needs far fewer discordant pairs than comparing
    two independent intervals would.
    """
    _check_universe(outcomes_a, gold)
    _check_universe(outcomes_b, gold)
    ids = list(gold) if pair_ids is None else list(pair_ids)
    out: dict[str, Any] = {}
    for rel, name in (("different", "fcr"), ("same", "mcr")):
        counts: Counter = Counter()
        for p in ids:
            if gold[p].relation != rel:
                continue
            if rel == "different":
                err_a, err_b = outcomes_a[p] == "merged", outcomes_b[p] == "merged"
            else:
                err_a, err_b = outcomes_a[p] != "merged", outcomes_b[p] != "merged"
            counts["both" if err_a and err_b else "only_a" if err_a else "only_b" if err_b else "neither"] += 1
        n = sum(counts.values())
        cell: dict[str, Any] = {k: counts[k] for k in ("only_a", "only_b", "both", "neither")}
        cell.update(
            n=n,
            rate_a=(counts["only_a"] + counts["both"]) / n if n else None,
            rate_b=(counts["only_b"] + counts["both"]) / n if n else None,
            delta=(counts["only_b"] - counts["only_a"]) / n if n else None,
            p_mcnemar=mcnemar_exact(counts["only_a"], counts["only_b"]) if n else None,
        )
        out[name] = cell
    return out


def check_passthrough(rows: Iterable[Mapping[str, Any]], records: Sequence[Tier1Record]) -> None:
    """Contract Sec 3: raise ``GoldError`` if an entry's ``gold_entity_id`` is not the input's opaque id for its
    ``source_utterance_id``. Entries of utterances not in the input are ignored (as in ``pair_outcomes``)."""
    expected: dict[str, str] = {}
    for r in records:
        expected[r.utterance_a_id], expected[r.utterance_b_id] = r.opaque_entity_id_a, r.opaque_entity_id_b
    bad = []
    for row in rows:
        for side in ("entry_a", "entry_b"):
            e = row[side]
            u = e.get("source_utterance_id")
            if u in expected and e.get("gold_entity_id") != expected[u]:
                bad.append(f"{row.get('record_id')}: {side} {u} has gold_entity_id {e.get('gold_entity_id')!r}")
    if bad:
        raise GoldError(
            f"{len(bad)} entr(ies) whose gold_entity_id does not match their source utterance, e.g. {bad[:2]}"
        )


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
    """Overall rates plus the per-class / per-language / cell breakdown for one ``pairwise_eval`` file.

    Raises ``GoldError`` first if an entry's ``gold_entity_id`` contradicts the input (``check_passthrough``)."""
    rows = list(rows)
    check_passthrough(rows, records)
    outcomes = pair_outcomes(rows, records, no_entry_ids=no_entry_ids)
    return {"overall": rates(outcomes, gold), "breakdown": breakdown(outcomes, gold, by), "outcomes": outcomes}
