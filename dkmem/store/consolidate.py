"""The write path: retrieve candidates, ask the host, apply the DK-Mem gate, act.

``consolidate(store, entry, host, metric, config)`` writes one new entry:

1. In a gated mode (``lexicon`` / ``lexicon+llm``) an entry whose ``distinction`` is
   unavailable (``None``, i.e. no V4 extraction in ``lexicon+llm``) is **not written**:
   the write is logged as ``write_skipped``. There is no fallback to lexicon-only
   distinctions, and nothing is turned into a decision.
2. ``retrieve`` the top ``k`` active entries (``dkmem.store.retrieval``).
3. The **host** proposes ``merge`` / ``keep_both`` for every candidate (an LLM judge on the
   two stored texts, or ``sim > tau``). A host that gives no usable answer yields a
   ``host_failed`` comparison: it is logged, makes no pairwise row and is never a decision.
4. The gate is applied host-first (``dkmem.memory.gate.apply_gate``): it only changes a
   proposed merge/supersede -- to ``keep_both`` when the pair is incompatible, to
   ``link_unresolved`` when it is underdetermined. So a link exists only where the host
   would have merged. Mode ``off`` skips the gate.
5. One action: if any candidate's final decision is merge/supersede, the new entry is
   absorbed into the **highest-ranked** such candidate (single target; newer text replaces
   older, the old text goes to ``history``); otherwise the new entry is added. Candidates
   whose final decision is ``link_unresolved`` get an unresolved link to the surviving
   entry. Links are logging-only.

Every candidate comparison is kept in the ``write`` event (with the candidate as it was
when compared), so Team B's ``pairwise_eval.jsonl`` rows are derived from the events
(``dkmem.store.export``) and the store itself holds no gold label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from dkmem.backends.llm import GenerationParams
from dkmem.config import dkmem_gate_enabled, validate_dkmem_mode
from dkmem.memory.dkmem_extract import apply_lexicon
from dkmem.memory.extract import TextGenerator
from dkmem.memory.gate import DEFAULT_DISCRIMINATIVE_FEATURES, MERGING_DECISIONS, apply_gate
from dkmem.memory.judge import MERGE_JUDGE_V1, judge_pairs
from dkmem.memory.lexicon import Lexicon
from dkmem.memory.prompts import PromptTemplate
from dkmem.memory.schema import Extraction
from dkmem.memory.similarity import SimilarityMetric
from dkmem.pipeline.trace import lexicon_distinction
from dkmem.store.entry import StoredEntry
from dkmem.store.retrieval import retrieve
from dkmem.store.store import MemoryStore, StoreError

__all__ = [
    "HOST_DECISIONS",
    "HostQuery",
    "HostProposal",
    "Host",
    "JudgeHost",
    "CachingHost",
    "ThresholdHost",
    "ConsolidationConfig",
    "WriteResult",
    "entry_distinction",
    "consolidate",
]

HOST_DECISIONS = ("merge", "keep_both")


# --- hosts -------------------------------------------------------------------------


@dataclass(frozen=True)
class HostQuery:
    """One comparison put to the host: the existing entry's text, the new text, their score.

    The order matches the frozen A-D harness (earlier entry first), so the judge sees
    ``Memory A`` = the entry already stored and ``Memory B`` = the one being written.
    """

    candidate_id: str
    candidate_text: str
    new_text: str
    score: float


@dataclass(frozen=True)
class HostProposal:
    """The host's proposal; ``decision`` is ``None`` when it produced no usable answer."""

    decision: str | None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision is not None and self.decision not in HOST_DECISIONS:
            raise ValueError(f"host decision must be None or one of {HOST_DECISIONS}, got {self.decision!r}")


class Host(Protocol):
    """A merge mechanism: ``mechanism`` names it; ``propose`` answers every query, in order."""

    mechanism: str

    def propose(self, queries: Sequence[HostQuery]) -> list[HostProposal]: ...


class JudgeHost:
    """Configs A-C: ``merge_judge_v1`` on the two stored texts (one batched call per write)."""

    mechanism = "llm_judge"

    def __init__(
        self,
        generator: TextGenerator,
        params: GenerationParams | None = None,
        *,
        prompt: PromptTemplate = MERGE_JUDGE_V1,
    ) -> None:
        self.generator, self.params, self.prompt = generator, params, prompt

    def propose(self, queries: Sequence[HostQuery]) -> list[HostProposal]:
        if not queries:
            return []
        results = judge_pairs(
            [(q.candidate_text, q.new_text) for q in queries], self.generator, self.params, prompt=self.prompt
        )
        return [
            HostProposal(
                r.decision,
                {
                    "prompt_id": self.prompt.prompt_id, "prompt_sha256": self.prompt.sha256,
                    "raw_output": r.raw_output, "decision": r.decision, "reason": r.reason, "error": r.error,
                },
            )
            for r in results
        ]


class CachingHost:
    """Memoizes a host's proposals by ``(candidate_text, new_text)``.

    The judge sees only the two stored texts, so one answer serves every DK-Mem mode and every
    episode that asks the same question (the frozen harness likewise judges each pair once and
    shares it across modes). Failed answers are cached too: nothing is retried. ``prewarm``
    answers many pairs in one batched call, so a runner can batch across episodes -- the store
    itself writes one entry at a time. ``stats`` counts what was asked live and what was reused.
    """

    def __init__(self, host: Host) -> None:
        self.host = host
        self.mechanism = host.mechanism
        self._memo: dict[tuple[str, str], HostProposal] = {}
        self.stats = {"live": 0, "reused": 0, "prewarmed": 0}

    def _fill(self, pairs: Sequence[tuple[str, str]]) -> None:
        todo = list(dict.fromkeys(p for p in pairs if p not in self._memo))
        if not todo:
            return
        answers = self.host.propose([HostQuery("", a, b, 0.0) for a, b in todo])
        if len(answers) != len(todo):
            raise ValueError(f"host returned {len(answers)} proposals for {len(todo)} queries")
        self._memo.update(zip(todo, answers))

    def prewarm(self, pairs: Sequence[tuple[str, str]]) -> None:
        """Answer ``(candidate_text, new_text)`` pairs now, in input order, in one host call."""
        before = len(self._memo)
        self._fill(pairs)
        self.stats["prewarmed"] += len(self._memo) - before

    def propose(self, queries: Sequence[HostQuery]) -> list[HostProposal]:
        pairs = [(q.candidate_text, q.new_text) for q in queries]
        self.stats["reused"] += sum(1 for p in pairs if p in self._memo)
        before = len(self._memo)
        self._fill(pairs)
        self.stats["live"] += len(self._memo) - before
        return [self._memo[p] for p in pairs]


class ThresholdHost:
    """Config D: merge iff the retrieval score is strictly above ``tau`` (``sim > tau``)."""

    mechanism = "embedding_threshold"

    def __init__(self, tau: float) -> None:
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be a number in [0, 1], got {tau!r}")
        self.tau = float(tau)

    def propose(self, queries: Sequence[HostQuery]) -> list[HostProposal]:
        return [HostProposal("merge" if q.score > self.tau else "keep_both", {"tau": self.tau}) for q in queries]


# --- configuration -------------------------------------------------------------------


@dataclass(frozen=True)
class ConsolidationConfig:
    """``mode`` is the DK-Mem mode (``off`` / ``lexicon`` / ``lexicon+llm``); ``k`` the retrieval depth."""

    mode: str
    k: int = 5
    discriminative_features: frozenset = DEFAULT_DISCRIMINATIVE_FEATURES

    def __post_init__(self) -> None:
        validate_dkmem_mode(self.mode)
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be a positive int, got {self.k!r}")
        object.__setattr__(self, "discriminative_features", frozenset(self.discriminative_features))

    @property
    def gated(self) -> bool:
        return dkmem_gate_enabled(self.mode)


def entry_distinction(
    mode: str, *, utterance: str, language: str, lexicon: Lexicon, v4: Extraction | None = None
) -> dict[str, str] | None:
    """The distinction dict the gate sees for one entry in ``mode`` (same rules as the A-D harness).

    ``off``: ``{}`` (the gate is not applied). ``lexicon``: the lexicon on the utterance.
    ``lexicon+llm``: the lexicon first, the V4 model's own distinction as fallback
    (``apply_lexicon``); ``None`` when there is no V4 extraction -- the caller must then
    not write the entry (see ``consolidate``).
    """
    validate_dkmem_mode(mode)
    if mode == "off":
        return {}
    if mode == "lexicon":
        return lexicon_distinction(lexicon, utterance, language)
    if v4 is None:
        return None
    return dict(apply_lexicon(v4, lexicon, language).distinction)


# --- the write -------------------------------------------------------------------------


@dataclass(frozen=True)
class WriteResult:
    """What one ``consolidate`` call did: ``action`` is ``add``, ``merge`` or ``skipped``."""

    write_idx: int
    entry_id: str
    action: str
    target_id: str | None
    event: dict[str, Any]


Diagnoser = Callable[[StoredEntry, StoredEntry, str], "dict[str, Any] | None"]


def _snapshot(e: StoredEntry) -> dict[str, Any]:
    return {
        "entry_id": e.entry_id, "source_utterance_id": e.source_utterance_id, "language": e.language,
        "text": e.stored_text, "surface": e.surface,
        "distinction": None if e.distinction is None else dict(e.distinction),
    }


def consolidate(
    store: MemoryStore,
    entry: StoredEntry,
    host: Host,
    metric: SimilarityMetric,
    config: ConsolidationConfig,
    *,
    diagnoser: Diagnoser | None = None,
) -> WriteResult:
    """Write ``entry`` into ``store`` (see the module docstring). Returns what was done.

    ``diagnoser(candidate, new_entry, host_proposed)`` is an optional hook whose result is
    stored as each comparison's ``diagnosis`` (e.g. ``dkmem.store.diagnosis.pair_diagnoser``).
    """
    seq = store.next_write_index()
    entry.seq = seq
    base = {
        "write_idx": seq, "conv_id": store.conv_id, "mode": config.mode, "k": config.k,
        "entry": _snapshot(entry),
    }

    if config.gated and entry.distinction is None:
        event = store.record(
            {
                "event": "write_skipped", **base, "action": "skipped",
                "reason": f"no distinction available for the new entry in mode {config.mode!r}; not written",
            }
        )
        return WriteResult(seq, entry.entry_id, "skipped", None, event)

    candidates = retrieve(
        store, entry.stored_text, metric, config.k, exclude_utterance_ids={entry.source_utterance_id}
    )
    n_active_before = len(store.active())
    existing = [store.get(c.entry_id) for c in candidates]
    if config.gated and any(e.distinction is None for e in existing):
        raise StoreError("a gated store holds an entry without a distinction; it should never have been written")

    queries = [HostQuery(c.entry_id, e.stored_text, entry.stored_text, c.score) for c, e in zip(candidates, existing)]
    proposals = host.propose(queries) if queries else []
    if len(proposals) != len(queries):
        raise ValueError(f"host returned {len(proposals)} proposals for {len(queries)} queries")

    comparisons: list[dict[str, Any]] = []
    for cand, cand_entry, prop in zip(candidates, existing, proposals):
        comp: dict[str, Any] = {
            "rank": cand.rank, "candidate_id": cand.entry_id, "score": cand.score,
            "candidate": _snapshot(cand_entry),
            "host": {"mechanism": host.mechanism, "proposed": prop.decision, "detail": dict(prop.detail)},
            "status": "ok" if prop.decision is not None else "host_failed",
            "gate": None, "final": None, "diagnosis": None,
        }
        if prop.decision is not None:
            if diagnoser is not None:
                comp["diagnosis"] = diagnoser(cand_entry, entry, prop.decision)
            if config.gated:
                g = apply_gate(prop.decision, cand_entry.distinction, entry.distinction, config.discriminative_features)
                comp["gate"] = {
                    "applied": True, "distinction_a": dict(cand_entry.distinction),
                    "distinction_b": dict(entry.distinction), "compatibility": g.compatibility,
                    "proposed": g.proposed, "decision": g.decision, "vetoed": g.vetoed, "reason": g.reason,
                }
                comp["final"] = g.decision
            else:
                comp["gate"] = {"applied": False, "proposed": prop.decision, "decision": prop.decision}
                comp["final"] = prop.decision
        comparisons.append(comp)

    # Single target: the highest-ranked candidate whose final decision unifies (comparisons are in rank order).
    merge_comp = next((c for c in comparisons if c["final"] in MERGING_DECISIONS), None)
    link_comps = [c for c in comparisons if c["final"] == "link_unresolved"]

    overwritten = None
    target_members_before = None
    if merge_comp is not None:
        target = store.get(merge_comp["candidate_id"])
        target_members_before = [m["source_utterance_id"] for m in target.members]
        store.absorb(target.entry_id, entry, decision=merge_comp["final"])
        overwritten = dict(target.history[-1])
        survivor, action, target_id = target.entry_id, "merge", target.entry_id
    else:
        store.add(entry)
        survivor, action, target_id = entry.entry_id, "add", None

    links = []
    for c in link_comps:
        is_new = store.link(survivor, c["candidate_id"], seq=seq, reason=c["gate"]["reason"])
        links.append({"a": survivor, "b": c["candidate_id"], "new": is_new, "reason": c["gate"]["reason"]})

    event = store.record(
        {
            "event": "write", **base, "action": action, "target_id": target_id,
            "decision": None if merge_comp is None else merge_comp["final"],
            "target_members_before": target_members_before, "overwritten": overwritten,
            "candidates": comparisons, "links": links,
            "had_failures": any(c["status"] == "host_failed" for c in comparisons),
            "n_active_before": n_active_before, "n_active_after": len(store.active()),
            "n_links_after": len(store.links),
        }
    )
    return WriteResult(seq, entry.entry_id, action, target_id, event)
