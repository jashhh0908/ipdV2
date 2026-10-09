"""The Mem0 v1.0.11 baseline on the Tier 1 episodes, in the A-D harness's output formats.

Per sanctioned Tier 1 record a **fresh** ``Mem0Memory`` (``user_id`` = ``eval_pair_id``): ``add(utterance_a)``,
then ``add(utterance_b)``; ``dkmem.baselines.mem0.events.classify_pair`` turns the events of b's call into one
pair outcome. Records run in input order, sequentially (Mem0 is sequential), with the run's decoding settings
(greedy, the same ``GenerationParams`` as Configs A-D), the same Qwen model and bge-m3, which is both Mem0's
embedder and the one similarity comparator (``cosine_metric``) used for the logged ``similarity_score``.

Outputs (``write_mem0_outputs``) -- the layout of an A-D group with a single ``off`` mode (no DK-Mem gate is
applied to Mem0; the gated variants of the same host idea are Configs A/B):

    <group>/run_config.json  trace.jsonl  summary.json
    <group>/off/run_manifest.json  pairwise_eval.jsonl      (strategy ``mem0``, no ``pipeline_config``)

``trace.jsonl`` (``dkmem_mem0_trace_v1``) keeps, for both calls, the raw LLM input and output of the extraction and
update steps, the retrieval hits, the id remapping, every action with its outcome, and the history rows, so a
decision can be audited down to the generated text. Failure policy is the A-D one: a side with no facts is
``no_entry``, an unusable update step is ``host_failed``, an unsupported language tag is skipped; none of them
becomes a decision or a ``pairwise_eval`` row.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dkmem.backends.embedding import cosine_metric
from dkmem.backends.llm import GenerationParams
from dkmem.baselines.mem0.events import MAPPING_ID, PAIR_OUTCOMES, classify_pair, mapping_rules_sha256
from dkmem.baselines.mem0.memory import SEARCH_LIMIT, BatchEmbedder, GeneratorLLM, Mem0AddError, Mem0Memory
from dkmem.baselines.mem0.source import PinnedMem0, load_pinned
from dkmem.memory.extract import TextGenerator, _parse_lang_tag
from dkmem.memory.similarity import SimilarityMetric
from dkmem.pipeline.runner import _backbone_slug, _count, _entry_id, _input_block, _skipped_row, mode_slug
from dkmem.pipeline.trace import config_fingerprint, git_state, inputs_hash, pair_input_hash, source_hashes
from dkmem.tier1.io import (
    DECISION_TRANSLATION,
    MemoryEntry,
    PairwiseEvalRecord,
    Tier1Record,
    write_pairwise_eval,
    write_run_manifest,
)

__all__ = [
    "MEM0_TRACE_SCHEMA",
    "MEM0_RUN_CONFIG_SCHEMA",
    "MAPPING_VERSION",
    "EMBEDDING_METRIC_NAME",
    "Mem0RunResult",
    "build_mem0_run_config",
    "run_mem0_baseline",
    "write_mem0_outputs",
]

MEM0_TRACE_SCHEMA = "dkmem_mem0_trace_v1"
MEM0_RUN_CONFIG_SCHEMA = "dkmem_mem0_run_config_v1"
MAPPING_VERSION = MAPPING_ID
EMBEDDING_METRIC_NAME = "bge_m3_dense_cosine_v1"
_STRATEGY = "mem0"
_MODE = "off"

_SUBSTITUTIONS = {
    "llm": "the run's Qwen generator (local Hugging Face), one prompt per call; response_format={'type': 'json_object'} "
           "is requested but cannot be enforced by a local model; the update call has no system message "
           "(the chat template adds its default)",
    "embedder": "bge-m3 dense CLS, L2-normalised (upstream default: OpenAI text-embedding-3-small)",
    "vector_store": "in-memory cosine search, exact-match filters (upstream default: Qdrant)",
    "history": "in-memory list with upstream's columns (upstream: SQLite)",
    "ids_and_clock": "deterministic uuid5 ids and a logical clock; neither enters a decision",
    "prompt_date": "the date line of USER_MEMORY_EXTRACTION_PROMPT is frozen (upstream embeds datetime.now())",
}


@dataclass
class Mem0RunResult:
    run_config: dict[str, Any]
    rows: list[PairwiseEvalRecord]
    trace: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def rows_by_mode(self) -> dict[str, list[PairwiseEvalRecord]]:
        return {_MODE: self.rows}


# --- run configuration ---------------------------------------------------------------------------------


def build_mem0_run_config(
    records: Sequence[Tier1Record],
    *,
    generator: TextGenerator,
    params: GenerationParams,
    embedder_info: Mapping[str, Any] | None = None,
    input_info: Mapping[str, Any] | None = None,
    pinned: PinnedMem0 | None = None,
) -> dict[str, Any]:
    """The reproducibility record of a Mem0 run. The pinned upstream hashes, the frozen prompt date, the prompt
    hashes, the model, decoding parameters, embedder and input hashes are in the fingerprint; the group id is
    ``mem0-v1.0.11-<backbone>-s<seed>-<fingerprint8>``."""
    pinned = pinned or load_pinned()
    info = getattr(generator, "run_info", None)
    run_info = info(params) if callable(info) else {"backbone": generator.backbone}
    pair_hashes = [pair_input_hash(r.eval_pair_id, r.language, r.utterance_a, r.utterance_b) for r in records]
    deterministic = {
        "schema": MEM0_RUN_CONFIG_SCHEMA,
        "baseline": {"name": "mem0", "mapping": MAPPING_VERSION, "mapping_sha256": mapping_rules_sha256(),
                     "pair_outcomes": list(PAIR_OUTCOMES)},
        "pipeline_config": None,
        "dkmem_modes": [_MODE],
        "seed": params.seed,
        "backbone": generator.backbone,
        "model_run_info": run_info,
        "generation_params": asdict(params),
        "judge_generation_params": None,
        "mem0": {
            **pinned.provenance,
            "add_flow": "Memory.add(infer=True): extract facts -> per-fact search(limit=5, no threshold) -> id remap "
                        "-> update LLM -> sequential ADD/UPDATE/DELETE/NONE",
            "search_limit": SEARCH_LIMIT,
            "search_threshold": None,
            "substitutions": _SUBSTITUTIONS,
        },
        "prompts": {
            "mem0_fact_extraction": {"prompt_id": "mem0_v1.0.11_USER_MEMORY_EXTRACTION_PROMPT",
                                     "sha256": pinned.provenance["prompts"]["fact_extraction"]["sha256"]},
            "mem0_update_memory": {"prompt_id": "mem0_v1.0.11_DEFAULT_UPDATE_MEMORY_PROMPT",
                                   "sha256": pinned.provenance["prompts"]["update_memory"]["sha256"]},
        },
        "lexicon": None,
        "similarity_metric": EMBEDDING_METRIC_NAME,
        "embedder": dict(embedder_info) if embedder_info else None,
        "tau": None,
        "tau_decides_merges": False,
        "input": {
            "n_records": len(records),
            "inputs_sha256": inputs_hash(pair_hashes),
            **{k: v for k, v in (input_info or {}).items() if k != "source_path"},
        },
    }
    fingerprint = config_fingerprint(deterministic)
    group_id = f"mem0-v1.0.11-{_backbone_slug(deterministic['backbone'])}-s{params.seed}-{fingerprint[:8]}"
    return {
        **deterministic,
        "fingerprint": fingerprint,
        "group_id": group_id,
        "input_source_path": (input_info or {}).get("source_path"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(Path(__file__).resolve().parent),
        "code": source_hashes(),
    }


# --- the run ------------------------------------------------------------------------------------------------


def _add(memory: Mem0Memory, utterance: str, user_id: str) -> tuple[dict[str, Any], str | None]:
    """``add`` -> ``(trace, exception text or None)``; an upstream-style exception keeps its partial trace."""
    try:
        return memory.add(utterance, user_id=user_id).trace, None
    except Mem0AddError as e:
        return e.trace, str(e)


def run_mem0_baseline(
    records: Sequence[Tier1Record],
    run_config: Mapping[str, Any],
    *,
    generator: TextGenerator,
    params: GenerationParams,
    embed_texts: Callable[[Sequence[str]], list[list[float]]],
    similarity: SimilarityMetric | None = None,
    pinned: PinnedMem0 | None = None,
) -> Mem0RunResult:
    """Run the Mem0 v1.0.11 reproduction over ``records`` (see the module docstring).

    ``embed_texts`` is ``BgeM3Embedder.embed``. ``similarity`` defaults to the bge-m3 cosine over the same
    embeddings, so Mem0's retrieval and the logged ``similarity_score`` use one comparator.
    """
    pinned = pinned or load_pinned()
    group_id = run_config["group_id"]
    metric = similarity or cosine_metric(embed_texts, EMBEDDING_METRIC_NAME)
    ids = [r.eval_pair_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("eval_pair_id must be unique: it names the per-episode memory")

    rows: list[PairwiseEvalRecord] = []
    trace: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"episodes": len(records), "episodes_skipped_unsupported_language": 0}
    for record in records:
        try:
            _parse_lang_tag(record.language)
        except ValueError as e:
            summary["episodes_skipped_unsupported_language"] += 1
            row = _skipped_row(group_id, "mem0", record, "unsupported_language", str(e))
            row["schema"] = MEM0_TRACE_SCHEMA
            trace.append(row)
            continue

        llm = GeneratorLLM(generator, params)
        memory = Mem0Memory(llm, BatchEmbedder(embed_texts), namespace=record.eval_pair_id, pinned=pinned)
        trace_a, exc_a = _add(memory, record.utterance_a, record.eval_pair_id)
        a_ids = [r["memory_id"] for r in trace_a.get("actions", []) if r.get("outcome") == "created"]
        before_b = {m["id"]: m["memory"] for m in memory.get_all(record.eval_pair_id)}
        trace_b, exc_b = _add(memory, record.utterance_b, record.eval_pair_id)

        scores = {h["id"]: h["score"] for r in trace_b.get("retrieval", []) for h in r["hits"] if h["id"] in a_ids}
        pair = classify_pair(trace_a, trace_b, a_memory_ids=a_ids, retrieval_scores=scores)
        row: dict[str, Any] = {
            "schema": MEM0_TRACE_SCHEMA, "group_id": group_id, "config": "mem0", "status": "ok" if pair.outcome in
            ("merge", "supersede", "keep_both") else pair.outcome,
            "eval_pair_id": record.eval_pair_id, "language": record.language,
            "input": _input_block(record),
            "add_a": trace_a, "add_b": trace_b, "exceptions": {"a": exc_a, "b": exc_b},
            "pair": {
                "mapping": MAPPING_VERSION, "outcome": pair.outcome, "reason": pair.reason, "a_state": pair.a_state,
                "b_state": pair.b_state, "a_memory_id": pair.a_memory_id, "action_index": pair.action_index,
            },
            "similarity": None, "final": None,
            "memory_final": memory.get_all(record.eval_pair_id),
            "n_llm_calls": len(llm.calls),  # the raw I/O of each call is in add_a / add_b
        }
        if pair.outcome in ("merge", "supersede", "keep_both"):
            a_index = a_ids.index(pair.a_memory_id)
            b_index = 0
            if pair.outcome == "keep_both":
                b_created = [r["memory_id"] for r in trace_b["actions"] if r.get("outcome") == "created"]
                b_index = b_created.index(pair.b_memory_id)
            entry_a_id, entry_b_id = _entry_id(record.utterance_a_id, a_index), _entry_id(record.utterance_b_id, b_index)
            text_a, text_b = before_b[pair.a_memory_id], pair.entry_b_text
            score = metric(text_a, text_b)
            internal = pair.outcome
            decision = DECISION_TRANSLATION[internal]
            unifies = internal in ("merge", "supersede")
            out = PairwiseEvalRecord(
                record_id=f"{entry_a_id}::{entry_b_id}", run_id=f"{group_id}-{mode_slug(_MODE)}", strategy=_STRATEGY,
                entry_a=MemoryEntry(entry_a_id, record.utterance_a_id, record.opaque_entity_id_a, record.language,
                                    text_a, record.utterance_a, None),
                entry_b=MemoryEntry(entry_b_id, record.utterance_b_id, record.opaque_entity_id_b, record.language,
                                    text_b, record.utterance_b, None),
                decision=decision, similarity_score=score, threshold=None, compatibility=None,
                predicted_entity_id=entry_a_id if unifies else None,
                superseded_entry_id=entry_a_id if internal == "supersede" else None,
            )
            rows.append(out)
            row["record_id"] = out.record_id
            row["similarity"] = {"metric": metric.name, "score": score, "text_a": text_a, "text_b": text_b, "decisive": False}
            row["final"] = {_MODE: {"decision": internal, "pairwise_decision": decision, "record_id": out.record_id}}
        trace.append(row)

    summary["pair_outcomes"] = _count(trace, lambda t: (t.get("pair") or {}).get("outcome"))
    summary["rows_by_mode"] = {_MODE: len(rows)}
    summary["final_decisions"] = {_MODE: _count(trace, lambda t: ((t.get("final") or {}).get(_MODE) or {}).get("pairwise_decision"))}
    summary["side_states"] = {
        side: _count(trace, lambda t, s=side: (t.get("pair") or {}).get(f"{s}_state")) for side in ("a", "b")
    }
    summary["applied_events"] = _count_actions(trace)
    summary["llm_calls"] = sum(t.get("n_llm_calls", 0) for t in trace)
    return Mem0RunResult(dict(run_config), rows, trace, summary)


def _count_actions(trace: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in trace:
        for key in ("add_a", "add_b"):
            for a in (t.get(key) or {}).get("actions", []):
                label = f"{a.get('event')}:{a.get('outcome')}"
                counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


# --- output ---------------------------------------------------------------------------------------------------


def write_mem0_outputs(out_dir: str | Path, result: Mem0RunResult) -> dict[str, str]:
    """Write ``run_config.json``, ``trace.jsonl``, ``summary.json`` and ``off/{run_manifest.json,pairwise_eval.jsonl}``."""
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
        backbone=cfg["backbone"], created_at=cfg["created_at"], dkmem_mode=_MODE,
    )
    write_pairwise_eval(d / "pairwise_eval.jsonl", result.rows)
    paths[f"{_MODE}/pairwise_eval"] = str(d / "pairwise_eval.jsonl")
    return paths
