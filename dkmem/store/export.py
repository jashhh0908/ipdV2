"""Store events -> Team B ``pairwise_eval.jsonl`` rows, and the store's own logs.

Pairwise rows
-------------
``pairwise_records`` turns the ``write`` events of one store run into
``PairwiseEvalRecord`` rows in the existing schema, one per (new entry, candidate) comparison
that reached a decision. ``entry_a`` is the candidate (the entry already stored, as it was
when compared), ``entry_b`` the new entry, the same orientation as the A-D harness, so
``record_id`` is ``"<candidate>::<new>"``. Not turned into rows: comparisons whose host gave no
usable answer (``host_failed``) and writes that were skipped (``write_skipped``) -- a failure
is never a decision.

Gold entity ids are *not* in the store; the caller supplies ``gold`` (``source_utterance_id``
-> gold entity id, e.g. from the Tier 1 input) and every row copies them through, as the
harness does.

For a candidate that is a merged cluster the row describes the cluster's survivor: its
``source_utterance_id`` and gold are those of the entry that founded the cluster, while
``gloss``/``surface`` are the latest text. This only matters beyond Tier 1.

Store logs
----------
``write_store_log`` writes ``store_events.jsonl`` (every event, in order) and
``store_final.json`` (entries, links, counts) -- the inputs for FCR/MCR at the cluster level,
memory bloat (``stats``) and error analysis. FCR/MCR themselves are Team B's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from dkmem.config import validate_dkmem_mode
from dkmem.store.store import MemoryStore
from dkmem.tier1.io import DECISION_TRANSLATION, MemoryEntry, PairwiseEvalRecord, translate_distinction

__all__ = ["pairwise_records", "write_store_log", "write_store_logs"]


def _memory_entry(snap: Mapping[str, Any], gold: Mapping[str, str], method: str | None, gated: bool) -> MemoryEntry:
    uid = snap["source_utterance_id"]
    if uid not in gold:
        raise KeyError(f"no gold entity id for source utterance {uid!r}")
    return MemoryEntry(
        entry_id=snap["entry_id"], source_utterance_id=uid, gold_entity_id=gold[uid],
        language=snap["language"], gloss=snap["text"], surface=snap["surface"],
        distinction=translate_distinction(snap["distinction"], extraction_method=method) if gated else None,
    )


def pairwise_records(
    events: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    run_id: str,
    strategy: str,
    threshold: float | None,
    gold: Mapping[str, str],
) -> list[PairwiseEvalRecord]:
    """``PairwiseEvalRecord`` rows for the ``write`` events (see the module docstring).

    ``threshold`` is the tau actually used (Config D) or ``None`` (an LLM-judge host);
    ``mode`` is the DK-Mem mode the events were produced in.
    """
    validate_dkmem_mode(mode)
    gated = mode != "off"
    method = mode if gated else None
    rows = []
    for ev in events:
        if ev["event"] != "write":
            continue
        for comp in ev["candidates"]:
            if comp["final"] is None:
                continue
            decision = DECISION_TRANSLATION[comp["final"]]
            cand = _memory_entry(comp["candidate"], gold, method, gated)
            new = _memory_entry(ev["entry"], gold, method, gated)
            unifies = decision in ("merge", "supersede")
            rows.append(
                PairwiseEvalRecord(
                    record_id=f"{cand.entry_id}::{new.entry_id}", run_id=run_id, strategy=strategy,
                    entry_a=cand, entry_b=new, decision=decision, similarity_score=comp["score"],
                    threshold=threshold, compatibility=comp["gate"]["compatibility"] if gated else None,
                    predicted_entity_id=cand.entry_id if unifies else None,
                    superseded_entry_id=cand.entry_id if decision == "supersede" else None,
                )
            )
    return rows


def write_store_logs(out_dir: str | Path, stores: Iterable[MemoryStore]) -> dict[str, str]:
    """For a run over many episodes (one store each): ``store_events.jsonl`` (every store's events,
    in order; each event carries its ``conv_id``) and ``store_final.jsonl`` (one final-state
    snapshot per store)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events_path, final_path = out / "store_events.jsonl", out / "store_final.jsonl"
    stores = list(stores)
    with open(events_path, "w", encoding="utf-8", newline="\n") as f:
        for s in stores:
            for ev in s.events:
                f.write(json.dumps({"conv_id": s.conv_id, **ev}, ensure_ascii=False, allow_nan=False) + "\n")
    with open(final_path, "w", encoding="utf-8", newline="\n") as f:
        for s in stores:
            f.write(json.dumps(s.snapshot(), ensure_ascii=False, allow_nan=False) + "\n")
    return {"events": str(events_path), "final": str(final_path)}


def write_store_log(out_dir: str | Path, store: MemoryStore) -> dict[str, str]:
    """Write ``store_events.jsonl`` and ``store_final.json`` into ``out_dir``; returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events_path, final_path = out / "store_events.jsonl", out / "store_final.json"
    with open(events_path, "w", encoding="utf-8", newline="\n") as f:
        for ev in store.events:
            f.write(json.dumps(ev, ensure_ascii=False, allow_nan=False) + "\n")
    final_path.write_text(
        json.dumps(store.snapshot(), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    return {"events": str(events_path), "final": str(final_path)}
