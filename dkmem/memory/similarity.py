"""DK-Mem consolidation candidates and gloss similarity.

Completes the write-side pipeline's other half:

    ProbeItem -> (extract / dkmem_extract) -> Extraction -> similarity -> gate -> MergeEvent

This module supplies the "similarity" step and the glue that turns two
``Extraction`` records into a ``MergeEvent`` via the existing, unmodified
``dkmem.memory.gate``. It does not touch extraction, any future memory
store, or the gate itself.

Candidates (not retrieval)
--------------------------
``candidate_pairs`` answers "given the entries already extracted from one
utterance-a and one utterance-b, which pairs of them are eligible to be
compared at all?" -- the cross product of the two sides, excluding
same-side pairs (nothing to consolidate an entry against another entry from
its own utterance). This is deliberately *not* retrieval: it does not
search an arbitrary memory store or rank candidates by anything; it assumes
the two entry lists are already known (today's ``extract``/``dkmem_extract``
always produce exactly one entry per side, so the cross product is a single
pair; a future multi-entry-per-utterance extractor would make this matter).
Building or querying an actual memory store is out of scope here.

This mirrors the Tier 1 evaluation contract's own candidate rule (Team A
handoff Sec 3): "Only records linking an utterance_a_id entry with an
utterance_b_id entry of the same record are scored, in either orientation"
and "Same-utterance comparisons are allowed, but ignored" -- i.e. Team B's
contract already assumes exactly this a-vs-b, no same-side, candidate
shape; this function makes that shape explicit and reusable in code rather
than something a later batch runner would have to reconstruct.

Gloss similarity
-----------------
``gloss_similarity`` computes the **raw, pre-gating** similarity between two
already-normalized glosses (``Extraction.gloss`` -- "the normalized English
retrieval key", per its own docstring). "Pre-gating" matters: this value
must never be clipped, rounded, or overwritten by anything the compatibility
gate decides (Team A handoff Sec 5) -- ``evaluate_pair`` below computes it
and passes it to the gate unchanged, and it is exactly the ``sim`` that ends
up on the resulting ``MergeEvent``, which is what a later Tier 1 writer
would use as ``similarity_score`` in ``pairwise_eval.jsonl``.

The metric is a deterministic character-sequence ratio
(``difflib.SequenceMatcher``, stdlib, no model call) over the two glosses
after light text normalization (casefold, collapsed whitespace) -- the same
technique already used for ``surface_similarity`` in
``tests/fixtures/synthetic_fixtures.py``, now promoted to production code so
fixtures and the real pipeline share one implementation instead of two.
This is a deterministic stand-in, not the paper's intended embedding
similarity (e.g. bge-m3): it is free, reproducible, and enough to exercise
the full Extraction -> similarity -> gate -> MergeEvent flow end to end.
Swapping in a real embedding model later only requires replacing
``gloss_similarity``'s body -- callers (``evaluate_pair``, and anything
downstream) do not need to change, since they only see a
``(gloss_a, gloss_b) -> float`` contract.
"""

from __future__ import annotations

import difflib
import re
from typing import Sequence

from dkmem.memory.gate import DEFAULT_DISCRIMINATIVE_FEATURES, build_merge_event
from dkmem.memory.schema import Extraction, MergeEvent

__all__ = [
    "SimilarityError",
    "normalize_gloss",
    "gloss_similarity",
    "candidate_pairs",
    "evaluate_pair",
]

_WHITESPACE_RE = re.compile(r"\s+")


class SimilarityError(ValueError):
    """Two Extractions can't be validly compared (mismatched pair/run identity)."""


def normalize_gloss(gloss: str) -> str:
    """Casefold and collapse whitespace in a gloss for similarity comparison.

    This is text normalization for comparison only -- separate from, and in
    addition to, the semantic gloss normalization the extractor already
    performs (English translation, third person, etc.).
    """
    return _WHITESPACE_RE.sub(" ", gloss).strip().casefold()


def gloss_similarity(gloss_a: str, gloss_b: str) -> float:
    """Deterministic, symmetric similarity in [0.0, 1.0] between two glosses.

    ``difflib.SequenceMatcher.ratio()`` is not always exactly symmetric
    (its matching-block heuristics can favor whichever sequence is passed
    first), which would make this "similarity" depend on which Extraction
    happens to be ``a`` vs ``b`` -- an artifact, not a property a
    consolidation threshold should be sensitive to. Comparing the two
    normalized glosses in a fixed (sorted) order makes the result exactly
    symmetric regardless of call order.

    See the module docstring for the metric and why it is a stand-in for a
    real embedding similarity, not the metric itself.
    """
    x, y = sorted((normalize_gloss(gloss_a), normalize_gloss(gloss_b)))
    return difflib.SequenceMatcher(None, x, y).ratio()


def candidate_pairs(
    entries_a: Sequence[Extraction], entries_b: Sequence[Extraction]
) -> list[tuple[Extraction, Extraction]]:
    """All (entry_a, entry_b) pairs eligible for consolidation gating.

    Every entry in ``entries_a`` must have ``side == "a"``, every entry in
    ``entries_b`` must have ``side == "b"``, and all entries (both lists)
    must share one ``pair_id`` -- raises ``SimilarityError`` otherwise, since
    the Tier 1 contract treats a cross-episode pairing as something that
    "should not occur". Returns the cross product, in ``entries_a`` then
    ``entries_b`` order; empty if either list is empty.
    """
    all_entries = list(entries_a) + list(entries_b)
    if not all_entries:
        return []
    pair_ids = {e.pair_id for e in all_entries}
    if len(pair_ids) > 1:
        raise SimilarityError(f"entries span more than one pair_id: {sorted(pair_ids)}")
    bad_a = [e.side for e in entries_a if e.side != "a"]
    bad_b = [e.side for e in entries_b if e.side != "b"]
    if bad_a or bad_b:
        raise SimilarityError(
            f"entries_a must all have side='a' (got {bad_a}); "
            f"entries_b must all have side='b' (got {bad_b})"
        )
    return [(a, b) for a in entries_a for b in entries_b]


def evaluate_pair(
    entry_a: Extraction,
    entry_b: Extraction,
    tau: float,
    *,
    policy: str,
    discriminative_features=DEFAULT_DISCRIMINATIVE_FEATURES,
) -> MergeEvent:
    """Compute similarity, gate it, and return the resulting ``MergeEvent``.

    This is the ``Extraction -> similarity -> gate -> MergeEvent`` wiring:
    ``sim = gloss_similarity(entry_a.gloss, entry_b.gloss)`` is computed
    once and handed unchanged to ``dkmem.memory.gate.build_merge_event``
    along with ``entry_a``/``entry_b``'s own ``distinction`` dicts -- neither
    the gate nor anything else here modifies it, so it stays the raw,
    pre-gating value the module docstring describes.

    ``entry_a``/``entry_b`` must share ``backbone``, ``prompt_id`` and
    ``seed`` (both sides of one comparison came from the same run) --
    raises ``SimilarityError`` otherwise. ``policy`` names the merge
    strategy being evaluated (e.g. ``"dkmem_v1"``) and is not inherent to
    either Extraction, so it is always caller-supplied, matching
    ``build_merge_event``'s own parameter.
    """
    for field in ("backbone", "prompt_id", "seed"):
        va, vb = getattr(entry_a, field), getattr(entry_b, field)
        if va != vb:
            raise SimilarityError(f"entry_a.{field} ({va!r}) != entry_b.{field} ({vb!r})")
    if entry_a.pair_id != entry_b.pair_id:
        raise SimilarityError(
            f"entry_a.pair_id ({entry_a.pair_id!r}) != entry_b.pair_id ({entry_b.pair_id!r})"
        )

    sim = gloss_similarity(entry_a.gloss, entry_b.gloss)
    return build_merge_event(
        entry_a.pair_id,
        entry_a.distinction,
        entry_b.distinction,
        sim=sim,
        tau=tau,
        policy=policy,
        backbone=entry_a.backbone,
        seed=entry_a.seed,
        prompt_id=entry_a.prompt_id,
        discriminative_features=discriminative_features,
    )
