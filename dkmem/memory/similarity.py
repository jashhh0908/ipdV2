"""DK-Mem candidate pairs and pluggable text similarity.

This module owns only the "similarity" step of

    ProbeItem -> (extract / dkmem_extract) -> Extraction -> similarity -> gate -> MergeEvent

and imports nothing from the gate: the gate takes ``sim`` as a plain number
(``dkmem.memory.gate``), and ``dkmem.memory.consolidation`` is the one place
that wires a similarity metric into the gate. Swapping the placeholder metric
for a real embedding therefore means constructing a ``SimilarityMetric`` and
passing it in; neither the gate nor any caller of ``entry_similarity``
changes.

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
gate decides (Team A handoff Sec 5) -- ``dkmem.memory.consolidation.
evaluate_pair`` passes it to the gate unchanged, and it is exactly the ``sim``
that ends up on the resulting ``MergeEvent`` (and, in the Tier 1 runner, as
``similarity_score`` in ``pairwise_eval.jsonl``).

The metric is a deterministic character-sequence ratio
(``difflib.SequenceMatcher``, stdlib, no model call) over the two glosses
after light text normalization (casefold, collapsed whitespace) -- the same
technique already used for ``surface_similarity`` in
``tests/fixtures/synthetic_fixtures.py``, now promoted to production code so
fixtures and the real pipeline share one implementation instead of two.
This is a deterministic stand-in, not the paper's intended embedding
similarity (bge-m3): it is free, reproducible, and enough to exercise the
pipeline end to end. A real embedding metric is added by wrapping its
``(text_a, text_b) -> float`` function in a ``SimilarityMetric`` (see below);
this placeholder is just one such metric, ``DIFFLIB_RATIO``.

Pluggable metrics
-----------------
``SimilarityMetric`` pairs a ``name`` (so the metric used can be reported with
results) with a ``(text_a, text_b) -> float`` function, and checks
every score is finite and within [0, 1] -- the range ``pairwise_eval.schema.
json`` requires. An out-of-range score raises ``SimilarityError`` instead of
being clipped, since the reported score must stay raw: a metric such as cosine
similarity (range [-1, 1]) has to map itself into [0, 1] explicitly.
Metrics should be symmetric. ``entry_similarity`` applies a metric to one text
field of two ``Extraction`` s (``gloss`` by default; ``surface`` for
configurations that compare verbatim text).
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from dkmem.memory.schema import Extraction

__all__ = [
    "SimilarityError",
    "SimilarityMetric",
    "SIMILARITY_TEXT_FIELDS",
    "DIFFLIB_RATIO",
    "DEFAULT_SIMILARITY",
    "normalize_gloss",
    "gloss_similarity",
    "entry_similarity",
    "candidate_pairs",
]

SIMILARITY_TEXT_FIELDS = ("gloss", "surface")

_WHITESPACE_RE = re.compile(r"\s+")


class SimilarityError(ValueError):
    """A similarity can't be validly computed: mismatched pair/run identity, or
    a metric returned a score outside [0, 1]."""


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


@dataclass(frozen=True)
class SimilarityMetric:
    """A named text-similarity function returning a score in [0, 1].

    ``name`` identifies the metric (and, for a learned one, should include the
    model) so results can record exactly what produced ``sim``. Calling the
    metric validates the score and never clips it.
    """

    name: str
    fn: Callable[[str, str], float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("SimilarityMetric.name must be a non-empty string")
        if not callable(self.fn):
            raise TypeError("SimilarityMetric.fn must be callable")

    def __call__(self, text_a: str, text_b: str) -> float:
        score = self.fn(text_a, text_b)
        ok = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
        if not ok or not 0.0 <= score <= 1.0:
            raise SimilarityError(
                f"metric {self.name!r} returned {score!r}; scores must be finite and in [0, 1] "
                "(they are reported raw, never clipped)"
            )
        return float(score)


# The deterministic placeholder (see "Gloss similarity" above).
DIFFLIB_RATIO = SimilarityMetric("difflib_ratio_v1", gloss_similarity)

DEFAULT_SIMILARITY = DIFFLIB_RATIO


def entry_similarity(
    entry_a: Extraction,
    entry_b: Extraction,
    metric: SimilarityMetric = DEFAULT_SIMILARITY,
    text_field: str = "gloss",
) -> float:
    """``metric`` applied to ``text_field`` of two Extractions: the raw,
    pre-gating similarity. Nothing here depends on the gate."""
    if text_field not in SIMILARITY_TEXT_FIELDS:
        raise ValueError(f"text_field must be one of {SIMILARITY_TEXT_FIELDS}, got {text_field!r}")
    return metric(getattr(entry_a, text_field), getattr(entry_b, text_field))


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
