"""Register sensitivity of the DK-Mem gate, replayed offline from an A-D group's trace (no model, no rerun).

Why: the frozen gate (``dkmem.memory.scope.DISCRIMINATIVE_CLASSES`` = kinship + register) vetoes a merge on any
register difference (Hindi tu / tum / aap). Register marks how the speaker addresses the listener, not who the fact is
about, and in Tier 1 v2 every Hindi ``honorific_register`` pair is gold ``same``; so register vetoes on this data can
mostly add missed consolidations. The frozen gate stays the **primary** result (it was fixed before any run); this
module adds one pre-registered sensitivity analysis (``DKMEM_ANALYSIS_PLAN.md``), never a replacement:

=================  ==========================================================================================
feature set        what the replayed gate treats as cannot-link evidence
=================  ==========================================================================================
``frozen``         ``DEFAULT_DISCRIMINATIVE_FEATURES`` (kinship + register), exactly what the run used
``kinship_only``   kinship alone (register is still extracted and logged, but no longer vetoes)
=================  ==========================================================================================

How: an A-D group makes one host decision per pair and logs, per DK-Mem mode, the host's proposal and the
distinctions the gate saw (``trace.jsonl`` ``gate.<mode>``). Re-applying ``apply_gate`` with another feature set to
those logged values gives exactly the decision that gate would have made on the same extractions and host decision,
so the comparison is paired. ``replay_outcomes`` first replays the ``frozen`` set and requires it to reproduce the
logged outcome of every pair (``SensitivityError`` otherwise), so a log that cannot support the replay is refused,
not silently mis-scored. Only A-D groups qualify (one cross-utterance record per pair); store, Mem0 and flat-RAG
groups are refused.

Two reports:

- ``gate_diagnostics`` needs **no gold key**: per mode, the outcome mix (overall and per input language), how often
  the host proposed a merge, how often the gate vetoed it, and which feature alone would have caused each veto. All
  counts are aggregate; no pair ids, no labels.
- ``sensitivity_report`` needs the key: per gated mode and feature set, FCR / MCR (Wilson 95%) and the paired exact
  McNemar test against gate-off and (for ``kinship_only``) against ``frozen``, overall, per gold
  ``distinction_class``, and over all pairs except ``honorific_register``. Every number is labelled
  ``PROVISIONAL_LABEL``: Tier 1 v2 relations are LLM-annotated, not human gold.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from dkmem.eval.gold import GoldPair
from dkmem.eval.groups import GroupData
from dkmem.eval.metrics import OUTCOMES, check_passthrough, pair_outcomes, paired_difference, rates
from dkmem.memory.gate import DEFAULT_DISCRIMINATIVE_FEATURES, MERGING_DECISIONS, apply_gate
from dkmem.tier1.io import Tier1Record

__all__ = [
    "FEATURE_SETS",
    "EXCLUDED_CLASS_VIEW",
    "PROVISIONAL_LABEL",
    "SensitivityError",
    "replay_outcomes",
    "gate_diagnostics",
    "sensitivity_report",
]

FEATURE_SETS: dict[str, frozenset[str]] = {
    "frozen": frozenset(DEFAULT_DISCRIMINATIVE_FEATURES),
    "kinship_only": frozenset({"kinship"}),
}

# the gold class the separate "without it" view leaves out (contract Sec 8: confounded with language and relation)
EXCLUDED_CLASS_VIEW = "honorific_register"

PROVISIONAL_LABEL = "Tier 1 v2, LLM-annotated, provisional"

_PIPELINE_CONFIGS = ("A", "B", "C", "D")


class SensitivityError(ValueError):
    """The group cannot be replayed (not an A-D group, or its log does not reproduce its own outcomes)."""


def _gated_blocks(group: GroupData, mode: str):
    """``(eval_pair_id, gate block)`` for every decided pair the gate was applied to in ``mode``."""
    for t in group.trace:
        if t.get("status") != "ok":
            continue
        g = (t.get("gate") or {}).get(mode)
        if g is None or "skipped" in g or not g.get("applied"):
            continue
        yield t["eval_pair_id"], g


def _replay(group: GroupData, mode: str, features, base: Mapping[str, str]) -> dict[str, str]:
    out = dict(base)
    for pid, g in _gated_blocks(group, mode):
        final = apply_gate(g["proposed"], g["distinction_a"], g["distinction_b"], features).decision
        out[pid] = "merged" if final in MERGING_DECISIONS else "compared_not_merged"
    return out


def replay_outcomes(
    group: GroupData, mode: str, features, records: Sequence[Tier1Record]
) -> dict[str, str]:
    """Every pair's outcome had the gate in ``mode`` used ``features``; checked against the logged outcomes first."""
    if group.config_id not in _PIPELINE_CONFIGS:
        raise SensitivityError(f"group {group.group_id} is not an A-D group; the gate replay needs one decision per pair")
    logged = pair_outcomes(group.rows(mode), records, no_entry_ids=group.no_entry_ids())
    frozen = _replay(group, mode, FEATURE_SETS["frozen"], logged)
    bad = sorted(p for p in logged if frozen[p] != logged[p])
    if bad:
        raise SensitivityError(f"{group.group_id}/{mode}: replaying the frozen gate changes {len(bad)} logged outcome(s)")
    return _replay(group, mode, features, logged)


def _veto_cause(g: Mapping[str, Any]) -> str:
    """Which discriminative features would, each on its own, have changed the host's proposal: e.g. ``register``."""
    alone = [f for f in sorted(FEATURE_SETS["frozen"])
             if apply_gate(g["proposed"], g["distinction_a"], g["distinction_b"], frozenset({f})).vetoed]
    return "+".join(alone) or "none"


def gate_diagnostics(group: GroupData, records: Sequence[Tier1Record]) -> dict[str, Any]:
    """Key-free gate activity of one A-D group (see the module docstring). Aggregate counts only."""
    language = {r.eval_pair_id: r.language for r in records}
    report: dict[str, Any] = {"group_id": group.group_id, "config": group.config_id, "tau": group.run_config.get("tau"),
                              "gold_key_used": False, "modes": {}}
    for mode in group.modes:
        outcomes = pair_outcomes(group.rows(mode), records, no_entry_ids=group.no_entry_ids())
        by_lang: dict[str, Counter] = {}
        for pid, o in outcomes.items():
            by_lang.setdefault(language[pid], Counter())[o] += 1
        entry: dict[str, Any] = {
            "outcomes": {o: sum(1 for v in outcomes.values() if v == o) for o in OUTCOMES},
            "outcomes_by_language": {k: {o: c[o] for o in OUTCOMES} for k, c in sorted(by_lang.items())},
        }
        blocks = list(_gated_blocks(group, mode))
        if blocks:
            vetoed = [(pid, g) for pid, g in blocks if g["vetoed"]]
            kinship_only = sum(
                1 for _, g in blocks
                if apply_gate(g["proposed"], g["distinction_a"], g["distinction_b"], FEATURE_SETS["kinship_only"]).vetoed
            )
            entry["gate"] = {
                "pairs_gated": len(blocks),
                "host_proposed_merge": sum(1 for _, g in blocks if g["proposed"] in MERGING_DECISIONS),
                "vetoed": len(vetoed),
                "vetoed_to": dict(sorted(Counter(g["decision"] for _, g in vetoed).items())),
                "vetoed_by_cause": dict(sorted(Counter(_veto_cause(g) for _, g in vetoed).items())),
                "vetoed_by_language": dict(sorted(Counter(language[pid] for pid, _ in vetoed).items())),
                "vetoed_if_kinship_only": kinship_only,
            }
        report["modes"][mode] = entry
    return report


def _views(gold: Mapping[str, GoldPair]) -> dict[str, list[str]]:
    return {
        "all": list(gold),
        f"excluding_{EXCLUDED_CLASS_VIEW}": [p for p, g in gold.items() if g.distinction_class != EXCLUDED_CLASS_VIEW],
    }


def _by_class(gold: Mapping[str, GoldPair]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for p, g in gold.items():
        groups.setdefault(str(g.distinction_class), []).append(p)
    return dict(sorted(groups.items()))


def sensitivity_report(
    group: GroupData, gold: Mapping[str, GoldPair], records: Sequence[Tier1Record]
) -> dict[str, Any]:
    """Frozen vs kinship-only gate on one A-D group, paired against gate-off (needs the key; aggregate only)."""
    if "off" not in group.modes:
        raise SensitivityError(f"group {group.group_id} has no gate-off mode to pair against")
    for mode in group.modes:
        check_passthrough(group.rows(mode), records)
    views, classes = _views(gold), _by_class(gold)
    off = pair_outcomes(group.rows("off"), records, no_entry_ids=group.no_entry_ids())
    report: dict[str, Any] = {
        "label": PROVISIONAL_LABEL, "group_id": group.group_id, "config": group.config_id,
        "tau": group.run_config.get("tau"), "feature_sets": {k: sorted(v) for k, v in FEATURE_SETS.items()},
        "off": {v: rates(off, gold, ids) for v, ids in views.items()}, "modes": {},
    }
    for mode in group.modes:
        if mode == "off":
            continue
        replayed = {name: replay_outcomes(group, mode, fs, records) for name, fs in FEATURE_SETS.items()}
        per_set: dict[str, Any] = {}
        for name, outcomes in replayed.items():
            cell: dict[str, Any] = {}
            for v, ids in views.items():
                cell[v] = {"rates": rates(outcomes, gold, ids), "paired_vs_off": paired_difference(off, outcomes, gold, ids)}
                if name != "frozen":
                    cell[v]["paired_vs_frozen"] = paired_difference(replayed["frozen"], outcomes, gold, ids)
            cell["by_class"] = {
                c: {"rates": rates(outcomes, gold, ids), "paired_vs_off": paired_difference(off, outcomes, gold, ids)}
                for c, ids in classes.items()
            }
            per_set[name] = cell
        report["modes"][mode] = per_set
    return report
