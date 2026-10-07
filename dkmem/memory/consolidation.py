"""Glue between similarity and the gate: two Extractions -> one MergeEvent.

``dkmem.memory.similarity`` (metrics) and ``dkmem.memory.gate`` (the
compatibility gate) do not know about each other; this is the only module that
connects them. The similarity metric is a parameter, so replacing the
placeholder ``difflib`` ratio with a real embedding metric changes a call
site, not the gate or this function.

``evaluate_pair`` uses the standalone ``gate()`` policy (merge iff
``sim >= tau`` and compatible; see ``dkmem.memory.gate``, "Two entry points").
"""

from __future__ import annotations

from dkmem.memory.gate import DEFAULT_DISCRIMINATIVE_FEATURES, build_merge_event
from dkmem.memory.schema import Extraction, MergeEvent
from dkmem.memory.similarity import (
    DEFAULT_SIMILARITY,
    SimilarityError,
    SimilarityMetric,
    entry_similarity,
)

__all__ = ["evaluate_pair"]


def evaluate_pair(
    entry_a: Extraction,
    entry_b: Extraction,
    tau: float,
    *,
    policy: str,
    discriminative_features=DEFAULT_DISCRIMINATIVE_FEATURES,
    similarity: SimilarityMetric = DEFAULT_SIMILARITY,
    text_field: str = "gloss",
) -> MergeEvent:
    """Compute similarity, gate it, and return the resulting ``MergeEvent``.

    ``sim = entry_similarity(entry_a, entry_b, similarity, text_field)`` is
    computed once and handed unchanged to ``dkmem.memory.gate.
    build_merge_event`` along with the entries' own ``distinction`` dicts --
    neither the gate nor anything else here modifies it, so it stays the raw,
    pre-gating value.

    ``entry_a``/``entry_b`` must share ``backbone``, ``prompt_id`` and
    ``seed`` (both sides of one comparison came from the same run) --
    raises ``SimilarityError`` otherwise. ``policy`` names the merge
    strategy being evaluated (e.g. ``"dkmem_v1"``) and is not inherent to
    either Extraction, so it is always caller-supplied.
    """
    for field in ("backbone", "prompt_id", "seed"):
        va, vb = getattr(entry_a, field), getattr(entry_b, field)
        if va != vb:
            raise SimilarityError(f"entry_a.{field} ({va!r}) != entry_b.{field} ({vb!r})")
    if entry_a.pair_id != entry_b.pair_id:
        raise SimilarityError(
            f"entry_a.pair_id ({entry_a.pair_id!r}) != entry_b.pair_id ({entry_b.pair_id!r})"
        )

    sim = entry_similarity(entry_a, entry_b, similarity, text_field)
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
