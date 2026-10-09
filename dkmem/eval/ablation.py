"""DK-Mem ablation: lexicon-only vs lexicon + LLM, on one host, one lexicon, one set of paired inputs.

DKMEM_NEW_RESEARCH_IDEA.md Sec 5.3 / 6.4: "Both variants are evaluated: lexicon-only (zero extra inference) and
lexicon + LLM", and "lexicon-only vs lexicon + LLM" is an ablation. In this repository both are DK-Mem *modes* of the
A-D harness (``dkmem.config.DKMEM_MODES``), run beside the gate-off host in the same group:

========================  ====================================================================================
mode                      distinctions the gate sees (then ``apply_gate`` vetoes the host's merge)
========================  ====================================================================================
``off``                   none; the host decides alone (the baseline for this host)
``lexicon``               the lexicon on the utterance only -- no model call for the gate
``lexicon+llm``           the lexicon first; the V4 model's distinction only for a class the lexicon flags as
                          ambiguous on that utterance (``resolve_distinction``); an extra V4 call in Configs B-D
                          only for such an utterance
========================  ====================================================================================

Because the host (judge answer or cosine) is made once per pair and shared, the three rows of a pair differ in
the gate only. ``ablation_report`` reads one group and returns, per mode, FCR/MCR (``dkmem.eval.metrics``), the gate's
activity (compatibility counts, vetoes, unresolved links, pairs with no decision), the cost in extra LLM calls, the
effect of the model fallback (pairs whose compatibility differs between ``lexicon`` and ``lexicon+llm``), and the
**pairing invariants**: every mode's row for a pair must carry the same similarity score, stored texts and
surface text; any difference is listed under ``pairing_violations``.

Specification note: Sec 5.3 (and research_idea_context.md) say "a small LLM call ... only for spans the lexicon flags
as ambiguous", and ``lexicon+llm`` implements exactly that (resolved 2026-10-08; before that the model was also used for
every class the lexicon did not cover). A class is ambiguous when the lexicon's own hits on the utterance give more than one
distinct value. A term the lexicon does not cover is therefore never tagged, so on an input with no ambiguous utterance
``lexicon+llm`` equals ``lexicon`` and ``extra_llm_calls`` is 0; ``fallback`` reports how often the model mattered.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from dkmem.eval.gold import GoldPair
from dkmem.eval.groups import GroupData
from dkmem.eval.metrics import check_passthrough, pair_outcomes, paired_difference, rates
from dkmem.tier1.io import Tier1Record

__all__ = ["ABLATION_MODES", "ablation_report"]

ABLATION_MODES = ("off", "lexicon", "lexicon+llm")

_ROW_FIELDS = ("similarity_score", "threshold")
_ENTRY_FIELDS = ("gloss", "surface", "source_utterance_id", "gold_entity_id", "language")


def _pairing_violations(group: GroupData) -> list[str]:
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    out = []
    for mode in group.modes:
        for r in group.rows(mode):
            sig = {f: r[f] for f in _ROW_FIELDS}
            for side in ("entry_a", "entry_b"):
                sig.update({f"{side}.{f}": r[side][f] for f in _ENTRY_FIELDS})
            first = seen.setdefault(r["record_id"], (mode, sig))
            if first[1] != sig:
                diff = sorted(k for k in sig if sig[k] != first[1][k])
                out.append(f"{r['record_id']}: {first[0]} vs {mode} differ in {diff}")
    return out


def _extra_llm_calls(group: GroupData, mode: str) -> int:
    """V4 calls the mode needs on top of the host: 0 for ``off`` / ``lexicon``; for ``lexicon+llm`` one per
    lexicon-ambiguous utterance (counted from the trace), 0 in Config A where the host's extraction is the V4 call."""
    if mode != "lexicon+llm" or group.config_id == "A":
        return 0
    n = 0
    for t in group.trace:
        for side in ("a", "b"):
            block = (t.get("extraction") or {}).get(side) or {}
            if block.get("model_distinction") is not None or block.get("v4_error") is not None:
                n += 1
    return n


def ablation_report(
    group: GroupData, gold: Mapping[str, GoldPair], records: Sequence[Tier1Record]
) -> dict[str, Any]:
    """The lexicon-only vs lexicon+LLM ablation of one A-D (or store) group; see the module docstring."""
    modes = [m for m in ABLATION_MODES if m in group.modes]
    report: dict[str, Any] = {
        "group_id": group.group_id, "config": group.config_id, "tau": group.run_config.get("tau"),
        "lexicon_sha256": (group.run_config.get("lexicon") or {}).get("sha256"), "modes": {},
        "pairing_violations": _pairing_violations(group),
    }
    off_outcomes = None
    for mode in modes:
        check_passthrough(group.rows(mode), records)
        outcomes = pair_outcomes(group.rows(mode), records, no_entry_ids=group.no_entry_ids())
        if mode == "off":
            off_outcomes = outcomes
        entry: dict[str, Any] = {"rates": rates(outcomes, gold), "extra_llm_calls": _extra_llm_calls(group, mode)}
        if mode != "off" and off_outcomes is not None:
            entry["paired_vs_off"] = paired_difference(off_outcomes, outcomes, gold)  # the gate's effect, pair by pair
        if mode != "off":
            gates = [((t.get("gate") or {}).get(mode), (t.get("final") or {}).get(mode)) for t in group.trace if t.get("status") == "ok"]
            decided = [(g, f) for g, f in gates if g is not None and "skipped" not in g]
            entry["gate"] = {
                "pairs_decided": len(decided),
                "pairs_without_decision": len(gates) - len(decided),
                "compatibility": {c: sum(1 for g, _ in decided if g["compatibility"] == c)
                                  for c in ("compatible", "incompatible", "underdetermined")},
                "vetoed_to_keep_both": sum(1 for g, f in decided if g["vetoed"] and f["decision"] == "keep_both"),
                "unresolved_links": sum(1 for _, f in decided if f["decision"] == "link_unresolved"),
            }
        report["modes"][mode] = entry

    if "lexicon" in modes and "lexicon+llm" in modes:
        changed, only_llm_blocks = [], 0
        for t in group.trace:
            if t.get("status") != "ok":
                continue
            gl, gm = (t["gate"] or {}).get("lexicon"), (t["gate"] or {}).get("lexicon+llm")
            if gl is None or gm is None or "skipped" in gl or "skipped" in gm:
                continue
            if gl["compatibility"] != gm["compatibility"]:
                changed.append({"pair": t["eval_pair_id"], "lexicon": gl["compatibility"], "lexicon+llm": gm["compatibility"]})
            for side in ("a", "b"):
                if gm[f"distinction_{side}"] != gl[f"distinction_{side}"]:
                    only_llm_blocks += 1
        report["fallback"] = {
            "pairs_whose_compatibility_changes": len(changed),
            "changes": changed,
            "utterances_whose_distinction_differs": only_llm_blocks,
            "delta_fcr": _delta(report, "fcr"), "delta_mcr": _delta(report, "mcr"),
        }
    return report


def _delta(report: Mapping[str, Any], key: str) -> float | None:
    a, b = report["modes"]["lexicon"]["rates"][key], report["modes"]["lexicon+llm"]["rates"][key]
    return None if a is None or b is None else b - a
