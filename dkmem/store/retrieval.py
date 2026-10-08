"""Candidate retrieval: the active entries most similar to a new entry's text.

Ranks the store's active entries by a pluggable ``SimilarityMetric`` applied to
``stored_text`` (the same text the judge sees and, in Config D, the embedding
compares) and returns the top ``k``. There is no score threshold here: in A-C the
judge decides each candidate, in D ``sim > tau`` does. Entries that already contain
content from one of the excluded source utterances (the new entry's own utterance)
are skipped, matching the Tier 1 rule that same-utterance comparisons are ignored.

The metric is deliberately not fixed: which one retrieves is a pre-registered choice
(the frozen A-D runs score A-C with the difflib stand-in and D with bge-m3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

from dkmem.memory.similarity import SimilarityMetric
from dkmem.store.store import MemoryStore

__all__ = ["Candidate", "retrieve"]


@dataclass(frozen=True)
class Candidate:
    """A retrieved entry: its id, 0-based ``rank`` (0 = most similar) and raw ``score``."""

    entry_id: str
    rank: int
    score: float


def retrieve(
    store: MemoryStore,
    text: str,
    metric: SimilarityMetric,
    k: int,
    *,
    exclude_utterance_ids: Collection[str] = (),
) -> list[Candidate]:
    """Top-``k`` active entries by ``metric(entry.stored_text, text)``, best first.

    Ties keep insertion order (older entry first), so ranking is deterministic.
    Returns fewer than ``k`` when the store holds fewer eligible entries.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}")
    excluded = set(exclude_utterance_ids)
    scored = [
        (metric(e.stored_text, text), e.entry_id)
        for e in store.active()
        if not (e.member_utterance_ids & excluded)
    ]
    scored.sort(key=lambda s: -s[0])  # stable: equal scores stay in insertion order
    return [Candidate(entry_id=eid, rank=i, score=score) for i, (score, eid) in enumerate(scored[:k])]
