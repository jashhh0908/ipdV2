"""Never-merge / flat dense RAG (DKMEM_NEW_RESEARCH_IDEA.md Sec 6.3): store every utterance, never consolidate.

The no-merge control that anchors the cost side of the frontier. It runs on the same infrastructure as the
consolidating systems -- ``MemoryStore``, ``consolidate`` (retrieve -> host -> single action), the bge-m3 cosine
comparator and the standard ``pairwise_eval`` export -- with a host that always answers ``keep_both``
(``NeverMergeHost``), so every write is an ``add`` and **pairwise FCR is 0 by construction** (``summary.json``
reports ``unifying_decisions: 0``; the tests assert it). It makes no LLM call, so ``backbone`` is ``null``; the DK-Mem
gate is not applied (``compatibility: null``, mode ``off``), since a gate cannot veto a merge that never happens.

Tier 1 (``run_flat_baseline``): per record a fresh store, utterance a then b, verbatim; one ``no_merge`` row per
episode with the bge-m3 cosine as the (informational) ``similarity_score`` and ``threshold: null``.

Tier 2 (``ingest_conversation``, ``FlatRAG``, ``retrieval_report``, ``answer_questions``): the same store holds a
whole conversation; questions are answered from the top-k stored utterances by the same retrieval function the
write path uses. The Tier 2 data does not exist yet, so the ``Turn`` / ``Question`` records here are a minimal,
provisional shape (``ingest_conversation`` is host-agnostic: pass a judge or threshold host to run a consolidating
system over the same conversation).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from dkmem.backends.embedding import cosine_metric
from dkmem.backends.llm import GenerationParams
from dkmem.memory.extract import TextGenerator, _parse_lang_tag
from dkmem.memory.prompts import PromptTemplate
from dkmem.memory.similarity import SimilarityMetric
from dkmem.pipeline.runner import _backbone_slug, _count, _entry_id, _input_block, _skipped_row, mode_slug
from dkmem.pipeline.trace import config_fingerprint, git_state, inputs_hash, pair_input_hash, source_hashes
from dkmem.store.consolidate import ConsolidationConfig, HostProposal, HostQuery, WriteResult, consolidate
from dkmem.store.entry import StoredEntry
from dkmem.store.export import pairwise_records, write_store_logs
from dkmem.store.retrieval import Candidate, retrieve
from dkmem.store.store import MemoryStore
from dkmem.tier1.io import Tier1Record, write_pairwise_eval, write_run_manifest

__all__ = [
    "FLAT_TRACE_SCHEMA",
    "FLAT_RUN_CONFIG_SCHEMA",
    "EMBEDDING_METRIC_NAME",
    "FLAT_QA_V1",
    "NeverMergeHost",
    "Turn",
    "Question",
    "QAResult",
    "FlatRunResult",
    "build_flat_run_config",
    "run_flat_baseline",
    "write_flat_outputs",
    "ingest_conversation",
    "FlatRAG",
    "retrieval_report",
    "render_qa_prompt",
    "answer_questions",
    "normalize_answer",
    "token_f1",
    "qa_report",
]

FLAT_TRACE_SCHEMA = "dkmem_flat_trace_v1"
FLAT_RUN_CONFIG_SCHEMA = "dkmem_flat_run_config_v1"
EMBEDDING_METRIC_NAME = "bge_m3_dense_cosine_v1"
_STRATEGY = "flat-dense-rag"
_MODE = "off"
DEFAULT_K = 5


class NeverMergeHost:
    """A host that never merges: every candidate is ``keep_both`` (so the store only ever ``add``s)."""

    mechanism = "never_merge"

    def propose(self, queries: Sequence[HostQuery]) -> list[HostProposal]:
        return [HostProposal("keep_both", {}) for _ in queries]


# --- Tier 1 ---------------------------------------------------------------------------------------------------


@dataclass
class FlatRunResult:
    run_config: dict[str, Any]
    rows: list[Any]
    trace: list[dict[str, Any]]
    summary: dict[str, Any]
    stores: list[MemoryStore]

    @property
    def rows_by_mode(self) -> dict[str, list[Any]]:
        return {_MODE: self.rows}


def build_flat_run_config(
    records: Sequence[Tier1Record],
    *,
    similarity_name: str = EMBEDDING_METRIC_NAME,
    embedder_info: Mapping[str, Any] | None = None,
    input_info: Mapping[str, Any] | None = None,
    k: int = DEFAULT_K,
    seed: int = 0,
) -> dict[str, Any]:
    """Reproducibility record of a flat-RAG run (no LLM: ``backbone`` is ``None``). Group id
    ``flat-no-llm-s<seed>-<fingerprint8>``."""
    pair_hashes = [pair_input_hash(r.eval_pair_id, r.language, r.utterance_a, r.utterance_b) for r in records]
    deterministic = {
        "schema": FLAT_RUN_CONFIG_SCHEMA,
        "baseline": {"name": _STRATEGY, "host": NeverMergeHost.mechanism, "storage": "verbatim utterance",
                     "episode_store": "fresh MemoryStore per episode; entry a written, then entry b; nothing consolidated"},
        "pipeline_config": None,
        "dkmem_modes": [_MODE],
        "seed": seed,
        "backbone": None,
        "model_run_info": None,
        "generation_params": None,
        "judge_generation_params": None,
        "prompts": {},
        "lexicon": None,
        "similarity_metric": similarity_name,
        "embedder": dict(embedder_info) if embedder_info else None,
        "k": k,
        "tau": None,
        "tau_decides_merges": False,
        "input": {
            "n_records": len(records),
            "inputs_sha256": inputs_hash(pair_hashes),
            **{key: v for key, v in (input_info or {}).items() if key != "source_path"},
        },
    }
    fingerprint = config_fingerprint(deterministic)
    return {
        **deterministic,
        "fingerprint": fingerprint,
        "group_id": f"flat-{_backbone_slug(None)}-s{seed}-{fingerprint[:8]}",
        "input_source_path": (input_info or {}).get("source_path"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(Path(__file__).resolve().parent),
        "code": source_hashes(),
    }


def run_flat_baseline(
    records: Sequence[Tier1Record],
    run_config: Mapping[str, Any],
    *,
    embed_texts: Callable[[Sequence[str]], list[list[float]]] | None = None,
    similarity: SimilarityMetric | None = None,
    k: int = DEFAULT_K,
) -> FlatRunResult:
    """Store both utterances of every record verbatim, never merging (see the module docstring)."""
    if similarity is None:
        if embed_texts is None:
            raise ValueError("run_flat_baseline needs embed_texts (bge-m3) or a similarity metric")
        similarity = cosine_metric(embed_texts, EMBEDDING_METRIC_NAME)
    group_id = run_config["group_id"]
    ids = [r.eval_pair_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("eval_pair_id must be unique: it names the per-episode store")
    host, ccfg = NeverMergeHost(), ConsolidationConfig(_MODE, k)

    stores: list[MemoryStore] = []
    trace: list[dict[str, Any]] = []
    skipped = 0
    gold: dict[str, str] = {}
    for record in records:
        try:
            _parse_lang_tag(record.language)
        except ValueError as e:
            skipped += 1
            row = _skipped_row(group_id, "flat", record, "unsupported_language", str(e))
            row["schema"] = FLAT_TRACE_SCHEMA
            trace.append(row)
            continue
        gold[record.utterance_a_id], gold[record.utterance_b_id] = record.opaque_entity_id_a, record.opaque_entity_id_b
        store = MemoryStore(record.eval_pair_id, meta={"baseline": _STRATEGY, "mode": _MODE, "k": k})
        writes = []
        for utt, uid in ((record.utterance_a, record.utterance_a_id), (record.utterance_b, record.utterance_b_id)):
            entry = StoredEntry(
                entry_id=_entry_id(uid, 0), conv_id=record.eval_pair_id, source_utterance_id=uid,
                language=record.language, stored_text=utt, surface=utt, distinction={},
            )
            writes.append(consolidate(store, entry, host, similarity, ccfg))
        stores.append(store)
        comp = writes[1].event["candidates"][0]
        trace.append({
            "schema": FLAT_TRACE_SCHEMA, "group_id": group_id, "config": "flat", "status": "ok",
            "eval_pair_id": record.eval_pair_id, "language": record.language,
            "record_id": f"{_entry_id(record.utterance_a_id, 0)}::{_entry_id(record.utterance_b_id, 0)}",
            "input": _input_block(record),
            "stored": {"a": {"entry_id": _entry_id(record.utterance_a_id, 0), "text": record.utterance_a},
                       "b": {"entry_id": _entry_id(record.utterance_b_id, 0), "text": record.utterance_b}},
            "similarity": {"metric": similarity.name, "score": comp["score"], "decisive": False},
            "host": {"mechanism": host.mechanism, "proposed": comp["host"]["proposed"]},
            "final": {_MODE: {"decision": comp["final"], "pairwise_decision": "no_merge",
                              "record_id": f"{comp['candidate_id']}::{writes[1].entry_id}"}},
            "store": {"actions": [w.action for w in writes], "entries_active": len(store.active())},
        })

    rows = pairwise_records(
        [ev for s in stores for ev in s.events], mode=_MODE, run_id=f"{group_id}-{mode_slug(_MODE)}",
        strategy=_STRATEGY, threshold=None, gold=gold,
    )
    unifying = sum(1 for r in rows if r.decision in ("merge", "supersede", "underdetermined_link"))
    summary = {
        "episodes": len(records), "episodes_skipped_unsupported_language": skipped, "pairs_ok": len(rows),
        "rows_by_mode": {_MODE: len(rows)}, "unifying_decisions": unifying,
        "final_decisions": {_MODE: _count(trace, lambda t: ((t.get("final") or {}).get(_MODE) or {}).get("pairwise_decision"))},
        "store": {
            "entries_active": sum(len(s.active()) for s in stores), "merges": sum(s.stats()["entries_absorbed"] for s in stores),
            "unresolved_links": sum(len(s.links) for s in stores),
        },
    }
    if unifying:
        raise AssertionError("flat RAG produced a unifying decision; it must never merge")
    return FlatRunResult(dict(run_config), rows, trace, summary, stores)


def write_flat_outputs(out_dir: str | Path, result: FlatRunResult) -> dict[str, str]:
    """``run_config.json``, ``trace.jsonl``, ``summary.json``, ``off/{run_manifest.json, pairwise_eval.jsonl,
    store_events.jsonl, store_final.jsonl}``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = result.run_config
    paths: dict[str, str] = {}
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    paths["run_config"] = str(out / "run_config.json")
    with open(out / "trace.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in result.trace:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    paths["trace"] = str(out / "trace.jsonl")
    (out / "summary.json").write_text(json.dumps(result.summary, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    paths["summary"] = str(out / "summary.json")
    d = out / mode_slug(_MODE)
    d.mkdir(exist_ok=True)
    write_run_manifest(
        d / "run_manifest.json", run_id=f"{cfg['group_id']}-{mode_slug(_MODE)}", strategy=_STRATEGY, seed=cfg["seed"],
        backbone=None, created_at=cfg["created_at"], dkmem_mode=_MODE,
    )
    write_pairwise_eval(d / "pairwise_eval.jsonl", result.rows)
    paths[f"{_MODE}/pairwise_eval"] = str(d / "pairwise_eval.jsonl")
    for name, path in write_store_logs(d, result.stores).items():
        paths[f"{_MODE}/store_{name}"] = path
    return paths


# --- Tier 2: conversations, retrieval, QA -----------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One user utterance of a conversation (provisional Tier 2 shape)."""

    utterance_id: str
    text: str
    language: str
    session_id: str | None = None
    turn_idx: int | None = None


@dataclass(frozen=True)
class Question:
    """A question with its gold answer and the utterances that carry the evidence (provisional Tier 2 shape)."""

    qid: str
    question: str
    answer: str
    evidence_utterance_ids: tuple[str, ...]


def ingest_conversation(
    store: MemoryStore,
    turns: Iterable[Turn],
    host: Any,
    metric: SimilarityMetric,
    config: ConsolidationConfig,
    *,
    diagnoser: Any = None,
) -> list[WriteResult]:
    """Write ``turns`` in order into ``store`` with ``consolidate`` -- the same write path as the Tier 1 runners.
    With ``NeverMergeHost`` this is the flat store; with a judge or threshold host it is a consolidating system
    over the same conversation. Turns are stored verbatim; in a gated ``config`` the caller must supply
    distinctions by building the entries itself and calling ``consolidate`` directly."""
    if config.gated:
        raise ValueError("ingest_conversation stores verbatim turns without distinctions; use mode 'off'")
    results = []
    for t in turns:
        entry = StoredEntry(
            entry_id=_entry_id(t.utterance_id, 0), conv_id=store.conv_id, source_utterance_id=t.utterance_id,
            language=t.language, stored_text=t.text, surface=t.text, distinction={},
            session_id=t.session_id, turn_idx=t.turn_idx,
        )
        results.append(consolidate(store, entry, host, metric, config, diagnoser=diagnoser))
    return results


class FlatRAG:
    """Flat dense RAG over one conversation: ``ingest`` verbatim turns, ``retrieve`` the top-k by the comparator."""

    def __init__(self, conv_id: str, metric: SimilarityMetric, *, k_write: int = DEFAULT_K) -> None:
        self.store = MemoryStore(conv_id, meta={"baseline": _STRATEGY})
        self.metric = metric
        self._host, self._cfg = NeverMergeHost(), ConsolidationConfig(_MODE, k_write)

    def ingest(self, turns: Iterable[Turn]) -> list[WriteResult]:
        return ingest_conversation(self.store, turns, self._host, self.metric, self._cfg)

    def retrieve(self, question: str, k: int) -> list[Candidate]:
        return retrieve(self.store, question, self.metric, k)


def retrieval_report(
    rag: FlatRAG, questions: Sequence[Question], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, Any]:
    """Evidence recall of the retrieval step: for each ``k``, the fraction of questions whose top-k contains
    *all* evidence utterances (``all_evidence``) and the mean fraction of evidence utterances found."""
    out: dict[str, Any] = {"n_questions": len(questions), "by_k": {}}
    for k in ks:
        full, frac = 0, 0.0
        for q in questions:
            found = set()
            for c in rag.retrieve(q.question, k):
                found |= rag.store.get(c.entry_id).member_utterance_ids
            hit = [u for u in q.evidence_utterance_ids if u in found]
            full += len(hit) == len(q.evidence_utterance_ids)
            frac += len(hit) / len(q.evidence_utterance_ids) if q.evidence_utterance_ids else 1.0
        out["by_k"][str(k)] = {
            "all_evidence": full / len(questions) if questions else None,
            "mean_evidence_fraction": frac / len(questions) if questions else None,
        }
    return out


_QA_SYSTEM = """\
You answer questions about a user from their stored memories. Use only the memories given. Answer in one short phrase or sentence in English. If the memories do not contain the answer, answer exactly: unknown"""

FLAT_QA_V1 = PromptTemplate(
    prompt_id="flat_qa_v1", system=_QA_SYSTEM, user_template="{utterance}", output_keys=(), examples=(),
)


def render_qa_prompt(question: str, memories: Sequence[str], prompt: PromptTemplate = FLAT_QA_V1) -> list[dict[str, str]]:
    block = "Memories:\n" + "\n".join(f"- {m}" for m in memories) + f"\n\nQuestion: {question}"
    return prompt.render(block)


@dataclass(frozen=True)
class QAResult:
    qid: str
    question: str
    gold: str
    answer: str
    retrieved: tuple[str, ...]
    retrieved_texts: tuple[str, ...]
    contains_gold: bool
    f1: float
    evidence_found: bool = field(default=False)


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.casefold()).split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_questions(
    rag: FlatRAG,
    questions: Sequence[Question],
    generator: TextGenerator,
    params: GenerationParams | None = None,
    *,
    k: int = DEFAULT_K,
    prompt: PromptTemplate = FLAT_QA_V1,
) -> list[QAResult]:
    """Answer ``questions`` from the top-``k`` retrieved memories, in one batched generator call."""
    retrieved = [rag.retrieve(q.question, k) for q in questions]
    texts = [[rag.store.get(c.entry_id).stored_text for c in cands] for cands in retrieved]
    outputs = generator.generate([render_qa_prompt(q.question, t, prompt) for q, t in zip(questions, texts)], params)
    if not isinstance(outputs, list) or len(outputs) != len(questions):
        raise ValueError(f"generator returned {len(outputs)} outputs for {len(questions)} questions")
    results = []
    for q, cands, tx, out in zip(questions, retrieved, texts, outputs):
        found = set()
        for c in cands:
            found |= rag.store.get(c.entry_id).member_utterance_ids
        results.append(QAResult(
            qid=q.qid, question=q.question, gold=q.answer, answer=out.strip(),
            retrieved=tuple(c.entry_id for c in cands), retrieved_texts=tuple(tx),
            contains_gold=normalize_answer(q.answer) in normalize_answer(out), f1=token_f1(out, q.answer),
            evidence_found=all(u in found for u in q.evidence_utterance_ids),
        ))
    return results


def qa_report(results: Sequence[QAResult]) -> dict[str, Any]:
    n = len(results)
    return {
        "n": n,
        "contains_gold": sum(r.contains_gold for r in results) / n if n else None,
        "mean_f1": sum(r.f1 for r in results) / n if n else None,
        "evidence_retrieved": sum(r.evidence_found for r in results) / n if n else None,
        "prompt_id": FLAT_QA_V1.prompt_id,
        "prompt_sha256": FLAT_QA_V1.sha256,
    }
