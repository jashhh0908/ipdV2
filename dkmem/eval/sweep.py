"""Raise-tau: the exact similarity-threshold sweep of Config D, computed offline from logged bge-m3 similarities.

The baseline of DKMEM_NEW_RESEARCH_IDEA.md Sec 6.3 ("the free, obvious fix"): make the embedding-threshold host
more conservative by raising tau, and see how much false consolidation it buys per unit of missed consolidation.
No model runs here. A Config D group logs, for every pair, the raw cosine of the two stored texts
(``similarity.score``) and, per DK-Mem mode, the distinctions the gate saw; the host's decision at any tau is
``merge`` iff ``score > tau`` (strictly greater, as ``ThresholdHost`` and the harness decide), and the DK-Mem gate
is applied on top with the same ``apply_gate`` the run used. So the whole FCR-MCR curve of every mode (``off`` is
the raise-tau baseline; ``lexicon`` / ``lexicon+llm`` are DK-Mem on the same host) comes from one run.

Exactness: the host decision vector only changes when tau crosses a logged score, so the grid is *every distinct
score* plus one point below the smallest (tau = -inf, ``tau: null``: everything proposed for merging). Nothing is
interpolated and no tau between two scores can give a result the grid lacks. ``self_check`` replays the run's own
tau and compares every pair's final decision with the logged one; a mismatch means the log and the sweep disagree.

Only Config D qualifies. A-C hosts are LLM judges whose merge decision uses no similarity cutoff; thresholding a
judge's token probabilities or logits is deliberately not implemented (``sweep_config_d`` refuses such a group).

Pairs without a decision in a mode (``lexicon+llm`` without a V4 extraction, unsupported language) stay *not
compared* at every tau, a constant MCR term, as the contract specifies.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from dkmem.eval.gold import GoldPair
from dkmem.eval.groups import GroupData
from dkmem.eval.metrics import rates
from dkmem.memory.gate import MERGING_DECISIONS, apply_gate

__all__ = [
    "SweepError",
    "tau_grid",
    "final_decision_at",
    "sweep_config_d",
    "pareto_frontier",
    "best_fcr_at_mcr",
    "compare_at_matched_mcr",
]

EPS = 1e-12


class SweepError(ValueError):
    """The group cannot be swept (not a threshold-decided run, or its log is incomplete)."""


def tau_grid(scores: Iterable[float]) -> list[float | None]:
    """Every distinct score ascending, preceded by ``None`` (= tau below the smallest score)."""
    return [None, *sorted(set(scores))]


def final_decision_at(score: float, tau: float | None, gate: Mapping[str, Any] | None) -> str | None:
    """The final internal decision of one pair at ``tau``: the host's ``sim > tau`` then the gate's veto.

    ``gate`` is the pair's logged gate block for the mode: ``{"applied": False, ...}`` for DK-Mem off (no veto),
    ``{"applied": True, "distinction_a", "distinction_b", ...}`` for a gated mode, or ``{"skipped": ...}``
    (-> ``None``: the pair has no decision in this mode).
    """
    if gate is None or "skipped" in gate:
        return None
    proposed = "merge" if (tau is None or score > tau) else "keep_both"
    if not gate.get("applied"):
        return proposed
    return apply_gate(proposed, gate["distinction_a"], gate["distinction_b"]).decision


def sweep_config_d(group: GroupData, gold: Mapping[str, GoldPair]) -> dict[str, Any]:
    """The exact sweep of a Config D group, for each of its DK-Mem modes (see the module docstring).

    Returns ``{"group_id", "run_tau", "modes": {mode: {"points": [...], "frontier": [...], "self_check": {...}}}}``.
    Each point: ``tau`` (``None`` = below the smallest score), ``fcr``, ``mcr``, their counts and denominators.
    """
    cfg = group.run_config
    if not cfg.get("tau_decides_merges"):
        raise SweepError(
            f"group {group.group_id} does not decide merges by a similarity threshold (host: LLM judge or other); "
            "only Config D is swept, and no probability/logit threshold is applied to a judge"
        )
    pairs = [t for t in group.trace if t.get("status") == "ok"]
    if any((t.get("similarity") or {}).get("score") is None for t in pairs):
        raise SweepError("a decided pair has no logged similarity score")
    all_ids = {t["eval_pair_id"] for t in group.trace}
    missing = sorted(set(gold) - all_ids)
    if missing:
        raise SweepError(f"{len(missing)} gold pair(s) are not in the group's trace, e.g. {missing[:3]}")
    grid = tau_grid(t["similarity"]["score"] for t in pairs)
    run_tau = cfg["tau"]

    modes: dict[str, Any] = {}
    for mode in group.modes:
        points = []
        for tau in grid:
            outcomes = {pid: "not_compared" for pid in all_ids}
            for t in pairs:
                final = final_decision_at(t["similarity"]["score"], tau, (t.get("gate") or {}).get(mode))
                if final is not None:
                    outcomes[t["eval_pair_id"]] = "merged" if final in MERGING_DECISIONS else "compared_not_merged"
            r = rates(outcomes, gold)
            points.append({
                "tau": tau, "fcr": r["fcr"], "mcr": r["mcr"], "false_merged": r["false_merged"], "missed": r["missed"],
                "n_different": r["n_different"], "n_same": r["n_same"],
            })
        mismatches = [
            t["eval_pair_id"] for t in pairs
            if final_decision_at(t["similarity"]["score"], run_tau, (t.get("gate") or {}).get(mode))
            != ((t.get("final") or {}).get(mode) or {}).get("decision")
        ]
        modes[mode] = {
            "points": points,
            "frontier": pareto_frontier(points),
            "self_check": {"tau": run_tau, "pairs_checked": len(pairs), "mismatches": mismatches},
        }
    return {"group_id": group.group_id, "run_tau": run_tau, "n_scores": len(grid) - 1, "modes": modes}


def pareto_frontier(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The non-dominated points (lower FCR and lower MCR are both better), ascending MCR.

    A point is dropped when another has FCR and MCR no larger and one strictly smaller. Points whose FCR or
    MCR is ``None`` (empty denominator) are skipped. Duplicates of the same (MCR, FCR) are listed once (the
    first, i.e. smallest tau)."""
    usable = [p for p in points if p["fcr"] is not None and p["mcr"] is not None]
    frontier = []
    for p in usable:
        dominated = any(
            q is not p and q["fcr"] <= p["fcr"] + EPS and q["mcr"] <= p["mcr"] + EPS
            and (q["fcr"] < p["fcr"] - EPS or q["mcr"] < p["mcr"] - EPS)
            for q in usable
        )
        if not dominated and not any(abs(f["fcr"] - p["fcr"]) < EPS and abs(f["mcr"] - p["mcr"]) < EPS for f in frontier):
            frontier.append(dict(p))
    return sorted(frontier, key=lambda p: (p["mcr"], p["fcr"]))


def best_fcr_at_mcr(points: Sequence[Mapping[str, Any]], mcr: float) -> dict[str, Any] | None:
    """The lowest-FCR point whose MCR is at most ``mcr`` (no interpolation); ``None`` if no point qualifies."""
    ok = [p for p in points if p["fcr"] is not None and p["mcr"] is not None and p["mcr"] <= mcr + EPS]
    return dict(min(ok, key=lambda p: (p["fcr"], p["mcr"]))) if ok else None


def compare_at_matched_mcr(
    system_points: Sequence[Mapping[str, Any]], raise_tau_points: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """For each point of a system (DK-Mem on D, Config A-C, Mem0, ...), the best raise-tau FCR at no more MCR.

    ``fcr_advantage`` = raise-tau FCR - system FCR: positive means the system beats raising tau at matched
    (or lower) MCR; ``None`` when raise-tau has no point with that little MCR (it cannot match the system)."""
    out = []
    for p in system_points:
        if p["fcr"] is None or p["mcr"] is None:
            continue
        ref = best_fcr_at_mcr(raise_tau_points, p["mcr"])
        out.append({
            "system": dict(p), "raise_tau": ref,
            "fcr_advantage": None if ref is None else ref["fcr"] - p["fcr"],
        })
    return out
