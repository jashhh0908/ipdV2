"""Mapping a Tier 1 episode's Mem0 events to one pair decision.

A Tier 1 episode is two ``Memory.add`` calls on a fresh memory: utterance a, then utterance b. The comparison
Team B scores is "did the system unify the entries from a and b?", but Mem0 never emits a per-pair decision --
it emits ADD / UPDATE / DELETE / NONE events for b's facts against whatever it retrieved (which, in a fresh
memory, can only be memories created by a). ``classify_pair`` turns those events into one of

======================  =====================================================================================
``supersede``           an applied UPDATE or DELETE hit a memory created from a: b's call overwrote or
                        removed a's memory (``pairwise_eval`` ``supersede``, ``superseded_entry_id`` = a's entry)
``keep_both``           b's call stored at least one new memory (applied ADD) and changed none of a's
``merge``               b's call stored nothing new and the update LLM answered NONE for a memory of a:
                        b's facts were judged to be already present
``no_entry``            an utterance produced zero facts (an empty list, or an output Mem0 could not parse and
                        silently turned into no facts): there is nothing to compare, no row is written
``host_failed``         the update step failed or was unusable (LLM exception, empty or unparseable response,
                        no ``memory`` list, or b's facts were neither stored nor matched to a memory of a)
======================  =====================================================================================

Precedence: ``no_entry`` (either side) > ``host_failed`` > ``supersede`` > ``keep_both`` > ``merge``. Only the
last three produce a ``pairwise_eval`` row; ``no_entry`` and ``host_failed`` are never decisions (as in the A-D
harness, where an unparsed extraction or judge answer yields a trace row and no pairwise row).

Judgement calls made here (they decide what counts as a false consolidation, so they are flagged in the
report): (1) UPDATE and DELETE both count as ``supersede``: either destroys or rewrites a's memory because of b.
Mem0 does not distinguish "richer description of the same entity" from "newer value", so a benign UPDATE is a
supersession too; (2) a NONE for a's memory alongside an ADD of b's fact is ``keep_both`` -- Mem0's own prompt
tells the LLM to list unrelated memories as NONE; (3) the first extracted fact of b stands for b's entry when b
stored nothing of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "PAIR_OUTCOMES",
    "SIDE_STATES",
    "MAPPING_ID",
    "MAPPING_RULES",
    "mapping_rules_sha256",
    "PairClassification",
    "side_state",
    "classify_pair",
]

MAPPING_ID = "mem0_pair_mapping_v1"

# FROZEN (2026-10-08, before any result was collected). The rules below are the evaluation contract of the Mem0
# baseline; ``classify_pair`` implements exactly them, and a test pins the sha256 of this text. Changing a rule
# means a new MAPPING_ID. Spec basis: DKMEM_NEW_RESEARCH_IDEA.md Sec 1 (a second fact that overwrites the first is
# the failure: "treat the second fact as an update ... the Pune fact is then deleted"), Sec 7 (FCR = pairs the
# system "merges or treats as a supersession") and tier1_eval_contract.md Sec 4 (a pair is merged iff its entries
# are connected by merge/supersede records; no_merge / underdetermined_link / not compared / no entry are not merged).
MAPPING_RULES = (
    "episode: fresh memory; add(utterance_a) then add(utterance_b); outcome is read from b's call",
    "no_entry: either utterance yields no facts (empty list, or output Mem0 cannot parse -> no facts); no row",
    "host_failed: a or b hit an LLM exception, or the update step was unusable (exception, empty/unparseable "
    "output, no 'memory' list), or a stored nothing, or b's facts were neither stored nor matched to a memory of a; no row",
    "supersede: an applied UPDATE or DELETE targets a memory created from a (UPDATE and DELETE both destroy or "
    "rewrite a because of b; Mem0 does not separate a richer description from a changed value); row decision "
    "'supersede', superseded_entry_id = a's entry; counts as unified (FCR / not an MCR miss)",
    "keep_both: no UPDATE/DELETE of a's memory and an applied ADD stored a new memory for b (a NONE for a's memory "
    "listed beside the ADD does not unify; Mem0's own prompt lists unrelated memories as NONE); row 'no_merge'",
    "merge: no UPDATE/DELETE, nothing stored for b, and an applied NONE maps to a memory of a (b's facts judged "
    "already present); row 'merge'",
    "precedence: no_entry > host_failed > supersede > keep_both > merge",
    "actions that failed to apply (hallucinated id, empty text, unknown event) are ignored",
    "entries: entry_a = the touched memory of a (keep_both: best retrieval score, ties earliest) with the text it "
    "held when b arrived; entry_b = b's new memory (keep_both) or b's first extracted fact (merge, supersede); "
    "similarity_score = the bge-m3 cosine of those two texts; threshold and compatibility null",
)


def mapping_rules_sha256() -> str:
    import hashlib

    return hashlib.sha256(chr(10).join(MAPPING_RULES).encode("utf-8")).hexdigest()

PAIR_OUTCOMES = ("merge", "supersede", "keep_both", "no_entry", "host_failed")
SIDE_STATES = (
    "ok", "no_facts", "extraction_parse_failed", "llm_exception", "update_failed", "nothing_stored",
)

_NO_ENTRY_STATES = ("no_facts", "extraction_parse_failed")


@dataclass(frozen=True)
class PairClassification:
    """The pair outcome and the entries a ``pairwise_eval`` row describes.

    ``a_memory_id`` / ``b_index`` are set for the three decisions only. ``entry_b_text`` is b's stored text
    (keep_both) or b's first extracted fact (merge, supersede).
    """

    outcome: str
    reason: str
    a_state: str
    b_state: str
    a_memory_id: str | None = None
    entry_b_text: str | None = None
    b_memory_id: str | None = None
    action_index: int | None = None


def side_state(trace: Mapping[str, Any] | None) -> str:
    """How far one ``add`` call got (``trace`` is ``AddResult.trace``; ``None`` when ``add`` raised before tracing)."""
    if trace is None:
        return "llm_exception"
    ext = trace.get("extraction") or {}
    if ext.get("llm_error") or trace.get("exception"):
        return "llm_exception"
    if ext.get("facts_error"):
        return "extraction_parse_failed"
    if not ext.get("facts"):
        return "no_facts"
    upd = trace.get("update") or {}
    parsed = upd.get("parsed")
    if upd.get("llm_error") or upd.get("parse_error") or not isinstance(parsed, dict) or not isinstance(parsed.get("memory"), list):
        return "update_failed"
    return "ok"


def _created(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [r for r in trace.get("actions", []) if r.get("outcome") == "created"]


def classify_pair(
    a_trace: Mapping[str, Any] | None,
    b_trace: Mapping[str, Any] | None,
    *,
    a_memory_ids: Sequence[str],
    retrieval_scores: Mapping[str, float] | None = None,
) -> PairClassification:
    """Classify one episode (see the module docstring). ``a_memory_ids`` are the memories a's call created.

    ``a_memory_ids`` is in creation order. ``retrieval_scores`` (memory id -> cosine with b's text) picks which
    of a's memories stands for a in a ``keep_both`` row (the best match; ties and no scores: the earliest one).
    """
    a_ids = set(a_memory_ids)
    a_state, b_state = side_state(a_trace), side_state(b_trace)
    if a_state == "ok" and not _created(a_trace):
        a_state = "nothing_stored"  # facts were extracted but the update step stored none of them

    if a_state in _NO_ENTRY_STATES or b_state in _NO_ENTRY_STATES:
        who = [s for s, st in (("a", a_state), ("b", b_state)) if st in _NO_ENTRY_STATES]
        return PairClassification("no_entry", f"no facts from {' and '.join(who)} ({a_state}/{b_state})", a_state, b_state)
    if a_state != "ok" or b_state != "ok":
        return PairClassification("host_failed", f"a: {a_state}; b: {b_state}", a_state, b_state)

    facts_b = b_trace["extraction"]["facts"]
    actions = b_trace.get("actions", [])
    hit = [r for r in actions if r.get("outcome") in ("updated", "deleted") and r.get("memory_id") in a_ids]
    if hit:
        r = hit[0]
        return PairClassification(
            "supersede", f"{r['event']} of a's memory {r['memory_id']}", a_state, b_state,
            a_memory_id=r["memory_id"], entry_b_text=facts_b[0], action_index=r["index"],
        )
    created = _created(b_trace)
    if created:
        scores = retrieval_scores or {}
        a_id = max(a_memory_ids, key=lambda m: scores.get(m, float("-inf")))  # first maximum: creation order
        return PairClassification(
            "keep_both", f"b stored {len(created)} new memor{'y' if len(created) == 1 else 'ies'}", a_state, b_state,
            a_memory_id=a_id, entry_b_text=created[0]["text"], b_memory_id=created[0]["memory_id"],
            action_index=created[0]["index"],
        )
    none_a = [r for r in actions if r.get("outcome") == "noop" and r.get("mapped_id") in a_ids]
    if none_a:
        r = none_a[0]
        return PairClassification(
            "merge", f"NONE for a's memory {r['mapped_id']}: b's facts already present", a_state, b_state,
            a_memory_id=r["mapped_id"], entry_b_text=facts_b[0], action_index=r["index"],
        )
    return PairClassification(
        "host_failed", "b's facts were neither stored nor matched to a memory of a", a_state, b_state
    )
