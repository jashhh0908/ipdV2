"""Per-episode Tier 1 runner for ``dk-mem-lexicon`` and ``dk-mem-lexicon-llm``.

Each of the 173 episodes is independent by construction: every call here
takes exactly one ``Tier1Record`` and returns a fresh result with no shared
mutable state, cache, or object carried in from a previous call. There is no
``ExtractionCache`` and no persistent store threaded across episodes -- the
only shared object is the (read-only, deterministic) ``Lexicon``, which is
reference data, not consolidated memory. This satisfies "reset memory
completely... nothing carries across episodes" by construction rather than
by an explicit reset step.

Two strategies (Task scope):

- ``dk-mem-lexicon``: ``extraction.extract_lexicon_only`` (no LLM call at
  all, ``backbone: None``).
- ``dk-mem-lexicon-llm``: the existing, unmodified
  ``dkmem.memory.dkmem_extract.dkmem_extract_many`` (Mem0-style baseline +
  lexicon overlay).

Both then run the identical translation: ``dkmem.memory.similarity.
candidate_pairs`` for the (always exactly one, today) a-vs-b comparison,
``dkmem.memory.similarity.entry_similarity`` for the raw pre-gating
similarity (the metric is a parameter, default the placeholder ``difflib``
ratio; a real embedding metric is passed in as ``similarity=``), and
``dkmem.memory.gate.gate`` (not ``evaluate_pair``/``build_merge_event``,
which don't expose the ``compatibility`` label Team B requires) for the
decision and compatibility. The standalone ``gate()`` policy is used, not
``apply_gate()``: there is no host merge mechanism in Tier 1 to veto.

The LLM strategy uses the default extraction prompt
(``dkmem.memory.prompts.DEFAULT_EXTRACTION_PROMPT``).

If extraction fails or raises for either side of an episode (a malformed
LLM response, or an unsupported ``language`` tag), that side simply yields
no entries: nothing is emitted for that episode, matching "if an utterance
yields no entries, emit nothing for it. Team B scores that pair as
'no entry'" -- not a crash of the whole run.
"""

from __future__ import annotations

from typing import Sequence

from dkmem.backends.llm import GenerationParams
from dkmem.memory.dkmem_extract import dkmem_extract_many
from dkmem.memory.extract import ExtractionError, TextGenerator
from dkmem.memory.gate import DEFAULT_DISCRIMINATIVE_FEATURES, gate
from dkmem.memory.lexicon import Lexicon
from dkmem.memory.schema import Extraction, ProbeItem
from dkmem.memory.similarity import (
    DEFAULT_SIMILARITY,
    SimilarityMetric,
    candidate_pairs,
    entry_similarity,
)
from dkmem.tier1.extraction import extract_lexicon_only
from dkmem.tier1.io import (
    DECISION_TRANSLATION,
    MemoryEntry,
    PairwiseEvalRecord,
    Tier1Record,
    translate_distinction,
)

__all__ = [
    "DEFAULT_TAU",
    "STRATEGY_EXTRACTION_METHOD",
    "run_episode_lexicon_only",
    "run_episode_lexicon_llm",
    "run_all_episodes_lexicon_only",
    "run_all_episodes_lexicon_llm",
]

# tau is only meaningful for one metric: 0.85 goes with the placeholder difflib
# ratio. A different similarity metric needs its own tau.
DEFAULT_TAU = 0.85

# strategy -> MemoryEntry.distinction.extraction_method (pairwise_eval.schema.json).
STRATEGY_EXTRACTION_METHOD = {
    "dk-mem-lexicon": "lexicon",
    "dk-mem-lexicon-llm": "lexicon+llm",
}


def _make_entry_id(utterance_id: str, index: int) -> str:
    return f"{utterance_id}_e{index}"


def _probe_item_adapter(record: Tier1Record) -> ProbeItem:
    """A ``ProbeItem`` shaped only to satisfy ``dkmem_extract_many``'s
    signature (it reads only ``pair_id``/``lang``/``utt_a``/``utt_b`` --
    nothing else). The gold-only fields below are inert placeholders:
    extraction never reads them, and nothing in ``dkmem.tier1`` reads them
    back off this object either -- the real output fields all come straight
    from ``record``.
    """
    return ProbeItem(
        pair_id=record.eval_pair_id,
        utt_a=record.utterance_a,
        utt_b=record.utterance_b,
        lang=record.language,
        distinction_class="tier1_unused",
        distinction_value_a=None,
        distinction_value_b=None,
        gold_same_entity=False,
        retrieval_query="",
    )


def _build_pairwise_records(
    record: Tier1Record,
    entries_a: Sequence[Extraction],
    entries_b: Sequence[Extraction],
    *,
    run_id: str,
    strategy: str,
    tau: float,
    similarity: SimilarityMetric = DEFAULT_SIMILARITY,
    discriminative_features=DEFAULT_DISCRIMINATIVE_FEATURES,
) -> list[PairwiseEvalRecord]:
    """Translate candidate (entry_a, entry_b) pairs into pairwise_eval rows."""
    extraction_method = STRATEGY_EXTRACTION_METHOD[strategy]
    pairs = candidate_pairs(entries_a, entries_b)
    entry_id_a = {id(e): _make_entry_id(record.utterance_a_id, i) for i, e in enumerate(entries_a)}
    entry_id_b = {id(e): _make_entry_id(record.utterance_b_id, i) for i, e in enumerate(entries_b)}

    rows = []
    for ext_a, ext_b in pairs:
        sim = entry_similarity(ext_a, ext_b, similarity)
        compatibility, internal_decision, _reason = gate(
            ext_a.distinction, ext_b.distinction, sim, tau, discriminative_features
        )
        decision = DECISION_TRANSLATION[internal_decision]

        entry_a = MemoryEntry(
            entry_id=entry_id_a[id(ext_a)],
            source_utterance_id=record.utterance_a_id,
            gold_entity_id=record.opaque_entity_id_a,
            language=record.language,
            gloss=ext_a.gloss,
            surface=ext_a.surface,
            distinction=translate_distinction(ext_a.distinction, extraction_method=extraction_method),
        )
        entry_b = MemoryEntry(
            entry_id=entry_id_b[id(ext_b)],
            source_utterance_id=record.utterance_b_id,
            gold_entity_id=record.opaque_entity_id_b,
            language=record.language,
            gloss=ext_b.gloss,
            surface=ext_b.surface,
            distinction=translate_distinction(ext_b.distinction, extraction_method=extraction_method),
        )
        # supersede is never produced by gate() today; merge shares entry_a's
        # id as the predicted entity (union-find-friendly: Team B unions on
        # entry_id).
        predicted_entity_id = entry_a.entry_id if decision in ("merge", "supersede") else None

        rows.append(
            PairwiseEvalRecord(
                record_id=f"{entry_a.entry_id}::{entry_b.entry_id}",
                run_id=run_id,
                strategy=strategy,
                entry_a=entry_a,
                entry_b=entry_b,
                decision=decision,
                similarity_score=sim,
                threshold=tau,
                compatibility=compatibility,
                predicted_entity_id=predicted_entity_id,
                superseded_entry_id=None,
            )
        )
    return rows


def run_episode_lexicon_only(
    record: Tier1Record,
    lexicon: Lexicon,
    *,
    run_id: str,
    tau: float = DEFAULT_TAU,
    seed: int = 0,
    similarity: SimilarityMetric = DEFAULT_SIMILARITY,
) -> list[PairwiseEvalRecord]:
    """One episode of ``dk-mem-lexicon`` (no LLM call)."""
    try:
        ext_a = extract_lexicon_only(
            record.eval_pair_id, "a", record.utterance_a, record.language, lexicon, seed=seed
        )
        ext_b = extract_lexicon_only(
            record.eval_pair_id, "b", record.utterance_b, record.language, lexicon, seed=seed
        )
    except ValueError:
        return []  # e.g. an unsupported language tag: no entries extracted
    return _build_pairwise_records(
        record, [ext_a], [ext_b], run_id=run_id, strategy="dk-mem-lexicon", tau=tau,
        similarity=similarity,
    )


def run_episode_lexicon_llm(
    record: Tier1Record,
    lexicon: Lexicon,
    generator: TextGenerator,
    params: GenerationParams,
    *,
    run_id: str,
    tau: float = DEFAULT_TAU,
    similarity: SimilarityMetric = DEFAULT_SIMILARITY,
) -> list[PairwiseEvalRecord]:
    """One episode of ``dk-mem-lexicon-llm`` (Mem0 baseline + lexicon overlay)."""
    probe = _probe_item_adapter(record)
    try:
        ext_a, ext_b = dkmem_extract_many(
            [(probe, "a"), (probe, "b")], generator, lexicon, params=params
        )
    except (ExtractionError, ValueError):
        return []
    return _build_pairwise_records(
        record, [ext_a], [ext_b], run_id=run_id, strategy="dk-mem-lexicon-llm", tau=tau,
        similarity=similarity,
    )


def run_all_episodes_lexicon_only(
    records: Sequence[Tier1Record],
    lexicon: Lexicon,
    *,
    run_id: str,
    tau: float = DEFAULT_TAU,
    seed: int = 0,
    similarity: SimilarityMetric = DEFAULT_SIMILARITY,
) -> list[PairwiseEvalRecord]:
    """Run every record independently; concatenate their pairwise rows, in order."""
    rows: list[PairwiseEvalRecord] = []
    for record in records:
        rows.extend(
            run_episode_lexicon_only(
                record, lexicon, run_id=run_id, tau=tau, seed=seed, similarity=similarity
            )
        )
    return rows


def run_all_episodes_lexicon_llm(
    records: Sequence[Tier1Record],
    lexicon: Lexicon,
    generator: TextGenerator,
    params: GenerationParams,
    *,
    run_id: str,
    tau: float = DEFAULT_TAU,
    similarity: SimilarityMetric = DEFAULT_SIMILARITY,
) -> list[PairwiseEvalRecord]:
    """Run every record independently; concatenate their pairwise rows, in order."""
    rows: list[PairwiseEvalRecord] = []
    for record in records:
        rows.extend(
            run_episode_lexicon_llm(
                record, lexicon, generator, params, run_id=run_id, tau=tau, similarity=similarity
            )
        )
    return rows
