"""The end-to-end store runner: Configs A-D through a real ``MemoryStore``.

    input -> extraction -> MemoryStore -> candidate retrieval -> host decision
          -> DK-Mem gate -> final memory state -> standardized outputs

``run_store_attribution`` is the store-based sibling of ``dkmem.pipeline.runner.
run_stage_attribution`` (which stays untouched). Same configurations
(A: V4 gloss + judge, B: native_b_v1 fact + judge, C: utterance + judge, D: utterance + bge-m3
``cosine > tau``), same DK-Mem modes (off / lexicon / lexicon+llm), same frozen prompts, same
extraction code (including its batch order), same output schemas -- but the merge decision
is taken by writing each stored entry into a ``MemoryStore`` with
``dkmem.store.consolidate.consolidate`` instead of comparing two entries directly.

How an episode runs
-------------------
Every Tier 1 episode is a tiny conversation: a **fresh** ``MemoryStore`` per (episode, DK-Mem
mode), the entry of utterance a written first, then utterance b's, so b's write retrieves a and
the host/gate decide. Stores are reset between episodes and modes; nothing carries over.

* Extraction runs once per side, batched across the whole input in the harness's order (V4 for
  Config A and for the ``lexicon+llm`` gate, ``native_b_v1`` for B, none for C/D). A side whose
  extraction fails stores nothing; the episode gets a trace row ``extraction_failed`` and no
  pairwise row, as in the frozen harness. The other side is still written (so its final state is
  visible) but is compared with nothing.
* The judge (A-C) is wrapped in ``CachingHost`` and pre-warmed with every pair in one batched call
  in input order -- the same batches, hence the same greedy answers, as the frozen harness; each
  text pair is judged once and shared by all modes. Config D's host is ``sim > tau`` on the
  retrieval score (bge-m3 cosine).
* **No fallback.** In ``lexicon+llm`` an entry without a V4 extraction is not written
  (``write_skipped``); the episode has no pairwise row in that mode. Judge failures are
  ``host_failed`` comparisons: logged, no row, never a decision.
* The retrieval metric is the harness's similarity: difflib for A-C unless one is passed, bge-m3
  cosine for D.

Outputs (``write_store_outputs``) -- everything ``dkmem.pipeline.runner.write_outputs`` writes
(``run_config.json``, ``trace.jsonl``, ``summary.json``, per mode ``run_manifest.json`` +
``pairwise_eval.jsonl``) plus, per mode, ``store_events.jsonl`` and ``store_final.jsonl`` (the
final memory state of every episode). The trace schema ``dkmem_store_trace_v1`` is
``dkmem_stage_trace_v1`` (``dkmem.pipeline.runner``) plus a ``store`` block per row (the write
actions, final entry and link counts per mode) and the status ``write_skipped``; rows are in input
order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dkmem.backends.llm import GenerationParams
from dkmem.config import get_pipeline_config
from dkmem.memory.cache import CacheKey, ExtractionCache
from dkmem.memory.dkmem_extract import ambiguous_distinction_classes
from dkmem.memory.extract import TextGenerator, _parse_lang_tag
from dkmem.memory.gate import DEFAULT_DISCRIMINATIVE_FEATURES
from dkmem.memory.judge import MERGE_JUDGE_V1
from dkmem.memory.lexicon import Lexicon
from dkmem.memory.native_extract import (
    NATIVE_B_V1,
    extraction_checks,
    native_fact_extract_many,
    output_language_report,
)
from dkmem.memory.prompts import DEFAULT_EXTRACTION_PROMPT, PromptTemplate
from dkmem.memory.similarity import DEFAULT_SIMILARITY, SimilarityMetric
from dkmem.pipeline.runner import (
    StageAttributionResult,
    _clean_modes,
    _count,
    _default_judge_params,
    _entry_id,
    _extract_tolerant,
    _extraction_block,
    _input_block,
    _native_summary,
    _probe_for,
    _Side,
    _skipped_row,
    _stored_block,
    _backbone_slug,
    _check_judge_prompt,
    _needs_v4_call,
    _episode_row,
    build_run_config,
    mode_slug,
    strategy_for,
    write_outputs,
)
from dkmem.pipeline.trace import config_fingerprint, lexicon_distinction
from dkmem.store.consolidate import (
    CachingHost,
    ConsolidationConfig,
    JudgeHost,
    ThresholdHost,
    WriteResult,
    consolidate,
    entry_distinction,
)
from dkmem.store.diagnosis import pair_diagnoser
from dkmem.store.entry import StoredEntry
from dkmem.store.export import pairwise_records, write_store_logs
from dkmem.store.store import MemoryStore
from dkmem.tier1.io import DECISION_TRANSLATION, Tier1Record

__all__ = [
    "STORE_TRACE_SCHEMA",
    "STORE_RUN_CONFIG_SCHEMA",
    "DEFAULT_K",
    "StoreRunResult",
    "build_store_run_config",
    "run_store_attribution",
    "write_store_outputs",
]

STORE_TRACE_SCHEMA = "dkmem_store_trace_v1"
STORE_RUN_CONFIG_SCHEMA = "dkmem_store_run_config_v1"
DEFAULT_K = 5

_OUTSIDE_FINGERPRINT = {"fingerprint", "group_id", "lexicon_path", "input_source_path", "created_at", "git", "code"}
_NO_V4 = "no V4 extraction for the model fallback"  # the harness's wording, kept so traces compare


@dataclass
class StoreRunResult(StageAttributionResult):
    """``StageAttributionResult`` plus the stores (one per episode and mode) the run produced."""

    stores_by_mode: dict[str, list[MemoryStore]]


# --- run configuration -----------------------------------------------------------------------


def build_store_run_config(
    config_id: str,
    dkmem_modes: Sequence[str],
    records: Sequence[Tier1Record],
    *,
    generator: TextGenerator | None,
    lexicon_path,
    params: GenerationParams,
    judge_params: GenerationParams | None = None,
    similarity: SimilarityMetric | None = None,
    tau: float | None = None,
    embedder_info: Mapping[str, Any] | None = None,
    input_info: Mapping[str, Any] | None = None,
    extraction_prompt: PromptTemplate = DEFAULT_EXTRACTION_PROMPT,
    k: int = DEFAULT_K,
    judge_prompt: PromptTemplate = MERGE_JUDGE_V1,
) -> dict[str, Any]:
    """The reproducibility record of a store run.

    The A-D harness's ``build_run_config`` (model, revision, seed, generation parameters, prompt
    hashes, lexicon hash, similarity, embedder, tau, input hash, per-file code hashes) with a
    ``store`` section added -- retrieval depth, retrieval metric, merge/link semantics -- inside
    the fingerprint, so a different ``k`` is a different run. The group id is
    ``store-<config>-<backbone>-s<seed>-<fingerprint8>``; the code hashes cover this package too.
    """
    base = build_run_config(
        config_id, dkmem_modes, records, generator=generator, lexicon_path=lexicon_path, params=params,
        judge_params=judge_params, similarity=similarity, tau=tau, embedder_info=embedder_info,
        input_info=input_info, extraction_prompt=extraction_prompt, judge_prompt=judge_prompt,
    )
    deterministic = {key: value for key, value in base.items() if key not in _OUTSIDE_FINGERPRINT}
    deterministic["schema"] = STORE_RUN_CONFIG_SCHEMA
    deterministic["store"] = {
        "runner": "dkmem_store_runner_v1",
        "k": k,
        "retrieval_metric": base["similarity_metric"],
        "discriminative_features": sorted(DEFAULT_DISCRIMINATIVE_FEATURES),
        "episode_store": "fresh MemoryStore per episode and DK-Mem mode; entry a written, then entry b",
        "merge": "single target: highest-ranked candidate that survives the gate; newer text replaces older, old text kept in history",
        "links": "unresolved link only when the host proposed a merge the gate changed to link_unresolved; logging-only",
        "failures": "no V4 fallback: lexicon+llm entry without V4 is not written; host failures are never decisions",
        "judge": f"{judge_prompt.prompt_id} once per stored-text pair, pre-warmed in input order, shared across modes",
    }
    fingerprint = config_fingerprint(deterministic)
    group_id = f"store-{config_id}-{_backbone_slug(deterministic['backbone'])}-s{params.seed}-{fingerprint[:8]}"
    return {
        **deterministic,
        "fingerprint": fingerprint,
        "group_id": group_id,
        **{key: base[key] for key in ("lexicon_path", "input_source_path", "created_at", "git", "code")},
    }


# --- extraction ----------------------------------------------------------------------------------


def _backend_info(generator: TextGenerator, params: GenerationParams) -> dict[str, Any] | None:
    """What identifies the model weights in an extraction-cache key (None for a generator without run_info)."""
    info = getattr(generator, "run_info", None)
    if not callable(info):
        return None
    run = info(params)
    cfg = run.get("model_config") or {}
    return {
        "model_id": cfg.get("model_id"), "revision": cfg.get("revision"), "dtype": cfg.get("dtype"),
        "resolved_revision": run.get("resolved_revision"),
    }


def _extract_v4(
    sides: list[_Side], generator: TextGenerator, params: GenerationParams, prompt: PromptTemplate,
    cache: ExtractionCache | None, stats: dict[str, int],
):
    """V4 extraction of every side, failures kept beside successes. With a cache, only misses are
    generated (successes are persisted; failures are not cached and would simply fail again)."""
    items = [(_probe_for(s.record), s.side) for s in sides]
    if cache is None:
        return _extract_tolerant(items, generator, params, prompt)
    info = _backend_info(generator, params)
    keys = [
        CacheKey.build(p, side, prompt=prompt, backbone=generator.backbone, params=params, backend_info=info)
        for p, side in items
    ]
    results: list[Any] = [None] * len(items)
    misses: dict[str, int] = {}
    for i, key in enumerate(keys):
        hit = cache.get(key)
        if hit is not None:
            results[i] = (hit, None)
            stats["hits"] += 1
        elif key.digest not in misses:
            misses[key.digest] = i
    idx = list(misses.values())
    generated = _extract_tolerant([items[i] for i in idx], generator, params, prompt)
    stats["generated"] += len(idx)
    by_digest = {}
    for i, (ext, err) in zip(idx, generated):
        by_digest[keys[i].digest] = (ext, err)
        if ext is not None:
            cache.put(keys[i], ext)
    return [r if r is not None else by_digest[keys[i].digest] for i, r in enumerate(results)]


# --- the run -----------------------------------------------------------------------------------------


def run_store_attribution(
    records: Sequence[Tier1Record],
    config_id: str,
    lexicon: Lexicon,
    run_config: Mapping[str, Any],
    *,
    generator: TextGenerator | None,
    params: GenerationParams,
    judge_params: GenerationParams | None = None,
    similarity: SimilarityMetric | None = None,
    tau: float | None = None,
    k: int = DEFAULT_K,
    extraction_prompt: PromptTemplate = DEFAULT_EXTRACTION_PROMPT,
    extraction_cache: ExtractionCache | None = None,
    judge_prompt: PromptTemplate = MERGE_JUDGE_V1,
) -> StoreRunResult:
    """Run one configuration over ``records`` through the memory store in every DK-Mem mode of
    ``run_config`` (``build_store_run_config``'s output for these same arguments).

    Config D requires ``similarity`` (the embedding metric) and ``tau``; the LLM-judge configs
    take no ``tau``. ``k`` is the retrieval depth (irrelevant to the result in Tier 1, where a
    store holds one entry when the second is written).
    """
    cfg = get_pipeline_config(config_id)
    modes = _clean_modes(run_config["dkmem_modes"])
    group_id = run_config["group_id"]
    embedding_host = cfg.merge_mechanism == "embedding_threshold"
    if embedding_host and (similarity is None or tau is None):
        raise ValueError("Config D needs an explicit similarity metric (bge-m3) and tau")
    if tau is not None and not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau!r}")
    if tau is not None and not embedding_host:
        raise ValueError(f"tau applies only to Config D; Config {config_id} merges with an LLM judge (no cutoff)")
    _check_judge_prompt(cfg, judge_prompt)
    metric = similarity or DEFAULT_SIMILARITY
    judge_params = judge_params or _default_judge_params(params)
    needs_v4 = cfg.extraction == "english_forced_prompt" or "lexicon+llm" in modes
    needs_generator = needs_v4 or cfg.extraction == "native_language_prompt" or cfg.merge_mechanism == "llm_judge"
    if needs_generator and generator is None:
        raise ValueError(f"Config {config_id} with modes {modes} needs a text generator")

    ids = [r.eval_pair_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("eval_pair_id must be unique: it names the per-episode store")
    summary: dict[str, Any] = {
        "episodes": len(records), "episodes_skipped_unsupported_language": 0,
        "sides_extraction_failed": 0, "pairs_judge_failed": 0, "pairs_ok": 0,
    }

    # --- sides ---------------------------------------------------------------------------------
    sides: list[_Side] = []
    skipped: dict[str, str] = {}
    for record in records:
        try:
            _parse_lang_tag(record.language)
        except ValueError as e:
            summary["episodes_skipped_unsupported_language"] += 1
            skipped[record.eval_pair_id] = str(e)
            continue
        for side, utt, uid in (("a", record.utterance_a, record.utterance_a_id),
                               ("b", record.utterance_b, record.utterance_b_id)):
            sides.append(_Side(
                record, side, utt, uid, lexicon_distinction(lexicon, utt, record.language),
                tuple(sorted(ambiguous_distinction_classes(lexicon, utt, record.language))),
            ))

    # --- extraction (same code and batch order as the frozen harness) ------------------------------
    cache_stats = {"hits": 0, "generated": 0}
    if needs_v4:
        v4_sides = [s for s in sides if _needs_v4_call(cfg, s)]
        for s, (ext, err) in zip(v4_sides, _extract_v4(v4_sides, generator, params, extraction_prompt, extraction_cache, cache_stats)):
            s.v4, s.v4_error = ext, err
    if cfg.extraction == "english_forced_prompt":
        for s in sides:
            if s.v4 is None:
                s.status, s.error = "extraction_failed", str(s.v4_error)
            else:
                s.entries = [(0, s.v4.gloss)]
    elif cfg.extraction == "native_language_prompt":
        for s, nat in zip(sides, native_fact_extract_many([s.utterance for s in sides], generator, params)):
            s.native = nat
            if not nat.ok:
                s.status, s.error = "extraction_failed", nat.error
            else:
                s.entries = [(0, nat.facts[0])]
                s.language_report = output_language_report(s.utterance, nat.facts[0], s.record.language)
                s.checks = extraction_checks(s.utterance, nat.facts[0])
    else:  # verbatim
        for s in sides:
            s.entries = [(0, s.utterance)]
    summary["sides_extraction_failed"] = sum(1 for s in sides if s.status == "extraction_failed")

    by_episode: dict[str, dict[str, _Side]] = {}
    for s in sides:
        by_episode.setdefault(s.record.eval_pair_id, {})[s.side] = s

    # --- host ------------------------------------------------------------------------------------------
    if cfg.merge_mechanism == "llm_judge":
        host: Any = CachingHost(JudgeHost(generator, judge_params, prompt=judge_prompt))
        host.prewarm(
            [
                (ta, tb)
                for record in records
                if (pair := by_episode.get(record.eval_pair_id)) is not None
                and pair["a"].status == "ok" and pair["b"].status == "ok"
                for _, ta in pair["a"].entries for _, tb in pair["b"].entries
            ]
        )
    else:
        host = ThresholdHost(tau)

    # --- episodes: a fresh store per episode and mode ----------------------------------------------------
    diagnoser = pair_diagnoser(lexicon, config_id)
    stores: dict[str, list[MemoryStore]] = {m: [] for m in modes}
    store_of: dict[str, dict[str, MemoryStore]] = {m: {} for m in modes}
    writes: dict[tuple[str, str], list[WriteResult]] = {}
    for mode in modes:
        ccfg = ConsolidationConfig(mode, k)
        for record in records:
            pair = by_episode.get(record.eval_pair_id)
            if pair is None:
                continue
            store = MemoryStore(record.eval_pair_id, meta={"config": config_id, "mode": mode, "tau": tau, "k": k})
            done: list[WriteResult] = []
            for s in (pair["a"], pair["b"]):
                if s.status != "ok":
                    continue
                for idx, text in s.entries:
                    entry = StoredEntry(
                        entry_id=_entry_id(s.utterance_id, idx), conv_id=record.eval_pair_id,
                        source_utterance_id=s.utterance_id, language=record.language,
                        stored_text=text, surface=s.utterance,
                        distinction=entry_distinction(
                            mode, utterance=s.utterance, language=record.language, lexicon=lexicon, v4=s.v4
                        ),
                    )
                    done.append(consolidate(store, entry, host, metric, ccfg, diagnoser=diagnoser))
            stores[mode].append(store)
            store_of[mode][record.eval_pair_id] = store
            writes[(mode, record.eval_pair_id)] = done

    # --- pairwise rows --------------------------------------------------------------------------------
    gold = {}
    for r in records:
        gold[r.utterance_a_id], gold[r.utterance_b_id] = r.opaque_entity_id_a, r.opaque_entity_id_b
    rows_by_mode = {
        mode: pairwise_records(
            [ev for s in stores[mode] for ev in s.events], mode=mode, run_id=f"{group_id}-{mode_slug(mode)}",
            strategy=strategy_for(config_id, mode, judge_prompt.prompt_id), threshold=tau, gold=gold,
        )
        for mode in modes
    }

    # --- trace -------------------------------------------------------------------------------------------
    trace: list[dict[str, Any]] = []
    for record in records:
        if record.eval_pair_id in skipped:
            row = _skipped_row(group_id, config_id, record, "unsupported_language", skipped[record.eval_pair_id])
            row["schema"] = STORE_TRACE_SCHEMA
            trace.append(row)
            continue
        sa, sb = by_episode[record.eval_pair_id]["a"], by_episode[record.eval_pair_id]["b"]
        row = _episode_trace_row(
            group_id, config_id, cfg, record, sa, sb, modes, writes, store_of, lexicon,
            tau=tau, metric_name=metric.name,
        )
        if row["status"] == "ok":
            summary["pairs_ok"] += 1
        elif row["status"] == "judge_failed":
            summary["pairs_judge_failed"] += 1
        trace.append(row)

    if cfg.extraction == "native_language_prompt":
        summary.update(_native_summary(sides))
    summary["rows_by_mode"] = {m: len(rows) for m, rows in rows_by_mode.items()}
    summary["first_loss_point"] = _count(trace, lambda t: (t.get("diagnosis") or {}).get("first_loss_point"))
    summary["host_proposed"] = _count(trace, lambda t: (t.get("host") or {}).get("proposed"))
    summary["final_decisions"] = {
        m: _count(trace, lambda t, m=m: ((t.get("final") or {}).get(m) or {}).get("pairwise_decision"))
        for m in modes
    }
    summary["store"] = {m: _store_summary(stores[m]) for m in modes}
    summary["host_cache"] = dict(host.stats) if isinstance(host, CachingHost) else None
    summary["extraction_cache"] = dict(cache_stats) if extraction_cache is not None else None
    return StoreRunResult(dict(run_config), rows_by_mode, trace, summary, stores)


def _store_summary(stores: list[MemoryStore]) -> dict[str, int]:
    """Bloat and action counts over all episodes of one mode."""
    totals = {"episodes": len(stores), "writes": 0, "writes_skipped": 0, "adds": 0, "merges": 0,
              "entries_active": 0, "unresolved_links": 0, "host_failures": 0}
    for s in stores:
        st = s.stats()
        totals["writes"] += st["writes"]
        totals["writes_skipped"] += st["writes_skipped"]
        totals["entries_active"] += st["entries_active"]
        totals["unresolved_links"] += st["unresolved_links"]
        for ev in s.events:
            if ev["event"] == "write":
                totals["adds" if ev["action"] == "add" else "merges"] += 1
                totals["host_failures"] += sum(1 for c in ev["candidates"] if c["status"] == "host_failed")
    return totals


def _episode_trace_row(
    group_id, config_id, cfg, record, sa: _Side, sb: _Side, modes, writes, stores, lexicon, *, tau, metric_name,
) -> dict[str, Any]:
    store_block = {}
    for mode in modes:
        s = stores[mode][record.eval_pair_id]
        store_block[mode] = {
            "actions": [w.action for w in writes[(mode, record.eval_pair_id)]],
            "entries_active": len(s.active()), "unresolved_links": len(s.links),
        }

    if sa.status == "extraction_failed" or sb.status == "extraction_failed":
        row = _episode_row(group_id, config_id, record, "extraction_failed", sa, sb, cfg, lexicon)
        row["schema"], row["store"] = STORE_TRACE_SCHEMA, store_block
        return row

    if len(sa.entries) != 1 or len(sb.entries) != 1:
        raise ValueError("the store runner expects exactly one stored entry per side (one fact per utterance)")
    entry_id_a, entry_id_b = _entry_id(sa.utterance_id, sa.entries[0][0]), _entry_id(sb.utterance_id, sb.entries[0][0])
    text_a, text_b = sa.entries[0][1], sb.entries[0][1]

    # the comparison of b's write against a, per mode (None if that mode skipped a write)
    comps: dict[str, dict[str, Any] | None] = {}
    for mode in modes:
        comps[mode] = None
        for w in writes[(mode, record.eval_pair_id)]:
            if w.event["event"] == "write" and w.event["candidates"]:
                comps[mode] = w.event["candidates"][0]
    canonical = next((comps[m] for m in modes if comps[m] is not None), None)

    status = "write_skipped" if canonical is None else "ok" if canonical["status"] == "ok" else "judge_failed"
    host_detail = None if canonical is None else canonical["host"]
    row: dict[str, Any] = {
        "schema": STORE_TRACE_SCHEMA, "group_id": group_id, "config": config_id, "status": status,
        "eval_pair_id": record.eval_pair_id, "language": record.language,
        "record_id": f"{entry_id_a}::{entry_id_b}",
        "input": _input_block(record),
        "extraction": {"a": _extraction_block(sa, cfg), "b": _extraction_block(sb, cfg)},
        "stored": {
            "a": _stored_block(entry_id_a, text_a, cfg, sa, record.language),
            "b": _stored_block(entry_id_b, text_b, cfg, sb, record.language),
        },
        "similarity": {
            "metric": metric_name, "score": None if canonical is None else canonical["score"],
            "text_a": text_a, "text_b": text_b, "decisive": cfg.merge_mechanism == "embedding_threshold",
        },
        "host": {
            "mechanism": cfg.merge_mechanism,
            "proposed": None if host_detail is None else host_detail["proposed"],
            "tau": tau, "tau_used": cfg.merge_mechanism == "embedding_threshold",
            "judge": host_detail["detail"] if host_detail is not None and cfg.merge_mechanism == "llm_judge" else None,
        },
        "gate": None, "final": None,
        "diagnosis": None if canonical is None else canonical["diagnosis"],
        "store": store_block,
    }
    if status == "ok":
        row["gate"], row["final"] = {}, {}
        for mode in modes:
            comp = comps[mode]
            if comp is None:
                row["gate"][mode] = {"applied": False, "skipped": _NO_V4}
                row["final"][mode] = None
            else:
                row["gate"][mode] = comp["gate"]
                row["final"][mode] = {
                    "decision": comp["final"], "pairwise_decision": DECISION_TRANSLATION[comp["final"]],
                    "record_id": row["record_id"],
                }
    return row


# --- output -----------------------------------------------------------------------------------------------


def write_store_outputs(out_dir, result: StoreRunResult) -> dict[str, str]:
    """``write_outputs`` of the A-D harness, plus per mode ``store_events.jsonl`` and
    ``store_final.jsonl`` (``<mode>/``). Returns the paths written, keyed by file."""
    from pathlib import Path

    paths = write_outputs(out_dir, result)
    for mode, stores in result.stores_by_mode.items():
        for name, path in write_store_logs(Path(out_dir) / mode_slug(mode), stores).items():
            paths[f"{mode}/store_{name}"] = path
    return paths
