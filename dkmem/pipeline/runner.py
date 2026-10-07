"""Stage-attribution harness: Configs A-D of DKMEM_NEW_RESEARCH_IDEA.md Sec 6.2.

One call, ``run_stage_attribution``, runs one configuration over a list of
Tier 1 records and applies the DK-Mem gate in each requested mode, so the
"gate off" and "gate on" results are paired on identical extractions, judge
calls and similarities.

======  ====================================  =====================  ==========
config  extraction -> stored text             host merge decision    isolates
======  ====================================  =====================  ==========
A       V4 prompt (English-forced) -> gloss   LLM judge on glosses   L1
B       native_b_v1 (keep user's language)    LLM judge on the fact  L2
C       none -> the utterance                 LLM judge on utterance L3
D       none -> the utterance                 bge-m3 cosine > tau    L4
======  ====================================  =====================  ==========

Per candidate pair (one entry from utterance a, one from utterance b; every
config gives exactly one pair per episode: Config B's frozen prompt extracts
exactly one fact per utterance) the harness records the chain *input -> extraction -> stored text ->
similarity -> host decision -> gate -> final decision* as one ``trace`` row
(``dkmem_stage_trace_v1``, below) and, per DK-Mem mode, one
``pairwise_eval.jsonl`` row in Team B's existing schema.

Design points
-------------
- The merge judge sees only the stored text (``dkmem.memory.judge``), so its
  decision does not depend on the gate. It is made once per pair and shared by
  all modes; the gate (``dkmem.memory.gate.apply_gate``) then vetoes a
  proposed merge when the distinctions are incompatible (-> ``keep_both``) or
  underdetermined (-> ``link_unresolved``). Config D proposes a merge when the
  similarity is strictly above tau (``sim > tau``, as in Sec 5.4 and the
  ``similarity_score`` description of ``pairwise_eval.schema.json``).
- DK-Mem distinctions never come from the host's extraction alone: mode
  ``lexicon`` uses the lexicon on the utterance; mode ``lexicon+llm`` adds the
  V4 model's own distinction as the fallback (``apply_lexicon``). In Config A
  that is the same V4 call that produced the gloss; in B/C/D a separate V4
  call is made, used for the gate's distinctions only (never stored or shown
  to the judge). Mode ``off`` uses none and reports ``compatibility: null``.
- Similarity is always computed on the stored text and logged. In Config D it
  decides (``sim > tau``); in A-C it is informational (the judge decides), its
  metric defaults to the difflib placeholder unless one is passed, and the output
  rows carry ``threshold: null`` and the run config ``tau: null`` /
  ``tau_decides_merges: false``: there is no cutoff, so none is invented.
- A failure is never turned into a decision: an extraction that cannot be
  parsed (including an empty fact), an episode with an unsupported language tag,
  or a judge answer that is not valid JSON yields a trace row with a ``status``
  other than ``ok`` and no ``pairwise_eval`` row for that pair, in every mode. A
  side whose extraction failed is reported as ``dropped_at_extraction`` in the
  row's ``diagnosis``.
- No gold label is read. ``opaque_entity_id_*`` is copied to ``gold_entity_id``
  on output only.

Trace row (``trace.jsonl``, one JSON object per candidate pair or skipped /
failed episode)::

    schema, group_id, config, status, eval_pair_id, language,
    input:        utterance_a/b, utterance_a/b_id, input_sha256
    extraction:   a, b -> mode, prompt_id, prompt_sha256, status, error,
                          raw_output, model_distinction, n_entries
    stored:       a, b -> entry_id, text, storage; Config B also output_language
                          (language drift / script drift) and checks (copy,
                          corruption signals), kept separate
    similarity:   metric, score, decisive
    host:         mechanism, proposed, tau, tau_used, judge (prompt, raw, ...)
    gate:         <mode> -> applied, distinction_a/b, compatibility, proposed,
                            decision, vetoed, reason
    final:        <mode> -> decision (internal), pairwise_decision, record_id
    diagnosis:    lexicon-relative loss point (dkmem.pipeline.trace.diagnose;
                  dropped_at_extraction for a side that stored nothing)

summary.json also reports, for Config B, ``stored_language`` (source language /
translated = language drift / script_changed = script drift / mixed) and
``extraction_checks`` (copies and automatic corruption signals). Corruption proper
(invented or changed content) needs reading the outputs; the signals are a lower bound.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dkmem.backends.llm import GenerationParams
from dkmem.config import get_pipeline_config, validate_dkmem_mode
from dkmem.memory.dkmem_extract import apply_lexicon
from dkmem.memory.extract import (
    ExtractionBatchError,
    ExtractionError,
    TextGenerator,
    _parse_lang_tag,
    extract_many,
)
from dkmem.memory.gate import apply_gate
from dkmem.memory.judge import MERGE_JUDGE_V1, JudgeResult, judge_pairs
from dkmem.memory.lexicon import Lexicon
from dkmem.memory.native_extract import (
    NATIVE_B_V1,
    NativeExtraction,
    extraction_checks,
    native_fact_extract_many,
    output_language_report,
)
from dkmem.memory.prompts import DEFAULT_EXTRACTION_PROMPT, PromptTemplate
from dkmem.memory.schema import Extraction, ProbeItem
from dkmem.memory.similarity import DEFAULT_SIMILARITY, SimilarityMetric
from dkmem.pipeline.trace import (
    config_fingerprint,
    diagnose,
    diagnose_dropped,
    git_state,
    inputs_hash,
    lexicon_distinction,
    pair_input_hash,
    sha256_file_lf,
    source_hashes,
)
from dkmem.tier1.io import (
    DECISION_TRANSLATION,
    MemoryEntry,
    PairwiseEvalRecord,
    Tier1Record,
    translate_distinction,
    write_pairwise_eval,
    write_run_manifest,
)

__all__ = [
    "TRACE_SCHEMA",
    "RUN_CONFIG_SCHEMA",
    "StageAttributionResult",
    "strategy_for",
    "mode_slug",
    "build_run_config",
    "run_stage_attribution",
    "write_outputs",
]

TRACE_SCHEMA = "dkmem_stage_trace_v1"
RUN_CONFIG_SCHEMA = "dkmem_run_config_v1"

_STORAGE_LOSS_STAGE = {"A": "L1", "B": "L2", "C": None, "D": None}
_HOST_LOSS_STAGE = {"A": "L3", "B": "L3", "C": "L3", "D": "L4"}


def strategy_for(config_id: str, dkmem_mode: str) -> str:
    """The ``strategy`` label of a (config, DK-Mem mode) run in the output schemas.

    Gate on: ``dk-mem-lexicon`` / ``dk-mem-lexicon-llm``. Gate off: the host
    role of Sec 6.3 -- ``mem0`` (Configs A, B), ``store-surface-only`` (C),
    ``embedding-threshold`` (D). ``pipeline_config`` in the manifest
    distinguishes A from B.
    """
    get_pipeline_config(config_id)
    validate_dkmem_mode(dkmem_mode)
    if dkmem_mode == "lexicon":
        return "dk-mem-lexicon"
    if dkmem_mode == "lexicon+llm":
        return "dk-mem-lexicon-llm"
    return {"A": "mem0", "B": "mem0", "C": "store-surface-only", "D": "embedding-threshold"}[config_id]


def mode_slug(dkmem_mode: str) -> str:
    return validate_dkmem_mode(dkmem_mode).replace("+", "-")


def _backbone_slug(backbone: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", backbone or "no-llm").strip("-").lower()


@dataclass
class StageAttributionResult:
    """Everything one ``run_stage_attribution`` call produced."""

    run_config: dict[str, Any]
    rows_by_mode: dict[str, list[PairwiseEvalRecord]]
    trace: list[dict[str, Any]]
    summary: dict[str, Any]


# --- run configuration --------------------------------------------------------


def build_run_config(
    config_id: str,
    dkmem_modes: Sequence[str],
    records: Sequence[Tier1Record],
    *,
    generator: TextGenerator | None,
    lexicon_path: str | Path,
    params: GenerationParams,
    judge_params: GenerationParams | None = None,
    similarity: SimilarityMetric | None = None,
    tau: float | None = None,
    embedder_info: Mapping[str, Any] | None = None,
    input_info: Mapping[str, Any] | None = None,
    extraction_prompt: PromptTemplate = DEFAULT_EXTRACTION_PROMPT,
) -> dict[str, Any]:
    """The reproducibility record of a run (model, seed, config, input hash).

    ``fingerprint`` is the sha256 of everything in the record that determines
    the outputs (not the creation time, git state, source hashes or file paths),
    and names the run group. ``input_info`` may carry ``source_path`` /
    ``file_sha256`` / ``sanctioned`` for the input file. ``code`` holds the sha256
    of every ``dkmem`` source/schema/lexicon file (line endings normalized) and a
    tree hash, so a result can be tied to the exact code even without git (as on
    Kaggle); it is deliberately outside the fingerprint so the run id does not
    change with a whitespace edit, but it is recorded with every run.

    ``tau`` applies only to Config D (embedding threshold); passing it for an
    LLM-judge config raises ``ValueError`` because there is no cutoff to set.
    """
    cfg = get_pipeline_config(config_id)
    if tau is not None and cfg.merge_mechanism != "embedding_threshold":
        raise ValueError(f"tau applies only to Config D; Config {config_id} merges with an LLM judge (no cutoff)")
    modes = _clean_modes(dkmem_modes)
    judge_params = judge_params or _default_judge_params(params)
    run_info = None
    if generator is not None:
        info = getattr(generator, "run_info", None)
        run_info = info(params) if callable(info) else {"backbone": generator.backbone}
    sim_metric = similarity or DEFAULT_SIMILARITY
    prompts = {}
    if cfg.extraction == "english_forced_prompt" or "lexicon+llm" in modes:
        prompts["distinction_extraction"] = _prompt_ref(extraction_prompt)
    if cfg.extraction == "native_language_prompt":
        prompts["native_extraction"] = _prompt_ref(NATIVE_B_V1)
    if cfg.merge_mechanism == "llm_judge":
        prompts["merge_judge"] = _prompt_ref(MERGE_JUDGE_V1)

    pair_hashes = [
        pair_input_hash(r.eval_pair_id, r.language, r.utterance_a, r.utterance_b) for r in records
    ]
    deterministic = {
        "schema": RUN_CONFIG_SCHEMA,
        "pipeline_config": asdict(cfg),
        "dkmem_modes": modes,
        "seed": params.seed,
        "backbone": generator.backbone if generator is not None else None,
        "model_run_info": run_info,
        "generation_params": asdict(params),
        "judge_generation_params": asdict(judge_params),
        "prompts": prompts,
        "lexicon": {"sha256": sha256_file_lf(lexicon_path)},
        "similarity_metric": sim_metric.name,
        "embedder": dict(embedder_info) if embedder_info else None,
        "tau": tau,
        "tau_decides_merges": cfg.merge_mechanism == "embedding_threshold",
        "input": {
            "n_records": len(records),
            "inputs_sha256": inputs_hash(pair_hashes),
            **{k: v for k, v in (input_info or {}).items() if k != "source_path"},
        },
    }
    fingerprint = config_fingerprint(deterministic)
    group_id = f"{config_id}-{_backbone_slug(deterministic['backbone'])}-s{params.seed}-{fingerprint[:8]}"
    return {
        **deterministic,
        "fingerprint": fingerprint,
        "group_id": group_id,
        "lexicon_path": str(lexicon_path),
        "input_source_path": (input_info or {}).get("source_path"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(Path(__file__).resolve().parent),
        "code": source_hashes(),
    }


def _prompt_ref(prompt: PromptTemplate) -> dict[str, str]:
    return {"prompt_id": prompt.prompt_id, "sha256": prompt.sha256}


def _clean_modes(modes: Sequence[str]) -> list[str]:
    out = []
    for m in modes:
        validate_dkmem_mode(m)
        if m not in out:
            out.append(m)
    if not out:
        raise ValueError("dkmem_modes must not be empty")
    return out


def _default_judge_params(params: GenerationParams) -> GenerationParams:
    return replace(params, max_new_tokens=min(params.max_new_tokens, 128))


# --- the run ------------------------------------------------------------------


def _probe_for(record: Tier1Record) -> ProbeItem:
    # ProbeItem is the input type of extract_many; only pair_id/lang/utt_* are read.
    return ProbeItem(
        pair_id=record.eval_pair_id, utt_a=record.utterance_a, utt_b=record.utterance_b,
        lang=record.language, distinction_class="tier1_unused", distinction_value_a=None,
        distinction_value_b=None, gold_same_entity=False, retrieval_query="",
    )


def _extract_tolerant(
    items: list[tuple[ProbeItem, str]], generator: TextGenerator, params: GenerationParams,
    prompt: PromptTemplate,
) -> list[tuple[Extraction | None, ExtractionError | None]]:
    """``extract_many`` that keeps successes and failures side by side."""
    if not items:
        return []
    try:
        return [(e, None) for e in extract_many(items, generator, params, prompt=prompt)]
    except ExtractionBatchError as err:
        failures = iter(err.failures)
        return [(e, None) if e is not None else (None, next(failures)) for e in err.extractions]


@dataclass
class _Side:
    record: Tier1Record
    side: str
    utterance: str
    utterance_id: str
    lex_dist: dict[str, str]
    v4: Extraction | None = None
    v4_error: ExtractionError | None = None
    native: NativeExtraction | None = None
    entries: list[tuple[int, str]] | None = None  # (index, stored text)
    status: str = "ok"
    error: str | None = None
    language_report: dict[str, Any] | None = None  # Config B: output_language_report of the fact
    checks: dict[str, Any] | None = None  # Config B: extraction_checks of the fact


@dataclass
class _Candidate:
    record: Tier1Record
    a: _Side
    b: _Side
    ia: int
    ib: int
    text_a: str
    text_b: str
    sim: float
    judge: JudgeResult | None = None
    proposed: str | None = None  # None -> the host produced no decision


def run_stage_attribution(
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
    extraction_prompt: PromptTemplate = DEFAULT_EXTRACTION_PROMPT,
) -> StageAttributionResult:
    """Run one configuration over ``records`` in every DK-Mem mode of ``run_config``.

    ``run_config`` is ``build_run_config``'s output for these same arguments
    (it names the run and fixes the modes). Config D requires ``similarity``
    (the embedding metric) and ``tau``. The LLM-judge configs take no ``tau``;
    their output rows carry ``threshold: null``.
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
    metric = similarity or DEFAULT_SIMILARITY
    threshold = tau  # None for the LLM-judge configs: there is no cutoff
    judge_params = judge_params or _default_judge_params(params)

    needs_v4 = cfg.extraction == "english_forced_prompt" or "lexicon+llm" in modes
    needs_generator = needs_v4 or cfg.extraction == "native_language_prompt" or cfg.merge_mechanism == "llm_judge"
    if needs_generator and generator is None:
        raise ValueError(f"Config {config_id} with modes {modes} needs a text generator")

    trace: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "episodes": len(records), "episodes_skipped_unsupported_language": 0,
        "sides_extraction_failed": 0, "pairs_judge_failed": 0, "pairs_ok": 0,
    }

    # --- episodes and sides --------------------------------------------------
    sides: list[_Side] = []
    for record in records:
        try:
            _parse_lang_tag(record.language)
        except ValueError as e:
            summary["episodes_skipped_unsupported_language"] += 1
            trace.append(_skipped_row(group_id, config_id, record, "unsupported_language", str(e)))
            continue
        for side, utt, uid in (("a", record.utterance_a, record.utterance_a_id),
                               ("b", record.utterance_b, record.utterance_b_id)):
            sides.append(_Side(record, side, utt, uid, lexicon_distinction(lexicon, utt, record.language)))

    # --- extraction -----------------------------------------------------------
    if needs_v4:
        results = _extract_tolerant(
            [(_probe_for(s.record), s.side) for s in sides], generator, params, extraction_prompt
        )
        for s, (ext, err) in zip(sides, results):
            s.v4, s.v4_error = ext, err

    if cfg.extraction == "english_forced_prompt":
        for s in sides:
            if s.v4 is None:
                s.status, s.error = "extraction_failed", str(s.v4_error)
            else:
                s.entries = [(0, s.v4.gloss)]
    elif cfg.extraction == "native_language_prompt":
        natives = native_fact_extract_many([s.utterance for s in sides], generator, params)
        for s, nat in zip(sides, natives):
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

    # --- candidate pairs and similarity ----------------------------------------
    by_episode: dict[str, dict[str, _Side]] = {}
    for s in sides:
        by_episode.setdefault(s.record.eval_pair_id, {})[s.side] = s
    candidates: list[_Candidate] = []
    for record in records:
        pair = by_episode.get(record.eval_pair_id)
        if pair is None:
            continue
        sa, sb = pair["a"], pair["b"]
        if sa.status == "extraction_failed" or sb.status == "extraction_failed":
            trace.append(_episode_row(group_id, config_id, record, "extraction_failed", sa, sb, cfg, lexicon))
            continue
        for ia, ta in sa.entries:
            for ib, tb in sb.entries:
                candidates.append(_Candidate(record, sa, sb, ia, ib, ta, tb, metric(ta, tb)))

    # --- host decision -----------------------------------------------------------
    if cfg.merge_mechanism == "llm_judge":
        judged = judge_pairs([(c.text_a, c.text_b) for c in candidates], generator, judge_params)
        for c, j in zip(candidates, judged):
            c.judge = j
            c.proposed = j.decision  # None when the answer was unusable
    else:
        for c in candidates:
            c.proposed = "merge" if c.sim > tau else "keep_both"

    # --- gate, rows, trace ---------------------------------------------------------
    rows_by_mode: dict[str, list[PairwiseEvalRecord]] = {m: [] for m in modes}
    for c in candidates:
        if c.proposed is None:
            summary["pairs_judge_failed"] += 1
        else:
            summary["pairs_ok"] += 1
        trace.append(
            _candidate_row(
                group_id, config_id, cfg, c, modes, lexicon, rows_by_mode, summary,
                threshold=threshold, tau=tau, metric_name=metric.name,
            )
        )
    if cfg.extraction == "native_language_prompt":
        summary.update(_native_summary(sides))
    summary["rows_by_mode"] = {m: len(rows) for m, rows in rows_by_mode.items()}
    summary["first_loss_point"] = _count(trace, lambda t: (t.get("diagnosis") or {}).get("first_loss_point"))
    summary["host_proposed"] = _count(trace, lambda t: (t.get("host") or {}).get("proposed"))
    summary["final_decisions"] = {
        m: _count(trace, lambda t, m=m: ((t.get("final") or {}).get(m) or {}).get("pairwise_decision"))
        for m in modes
    }
    return StageAttributionResult(dict(run_config), rows_by_mode, trace, summary)


def _native_summary(sides: list[_Side]) -> dict[str, Any]:
    """Config B: language drift, script drift and corruption signals, kept separate.

    Counted over every side that produced a fact (not only those that reached the
    judge). ``translated`` is language drift (the fact is not in the source
    language); ``script_changed`` is script drift (same language, other script).
    """
    ok = [s for s in sides if s.language_report is not None]
    labels = Counter(s.language_report["label"] for s in ok)
    checks = [s.checks for s in ok]
    return {
        "stored_language": {
            "entries": len(ok),
            "source_language": labels.get("source_language", 0),
            "translated": labels.get("translated_to_english", 0),
            "script_changed": labels.get("script_changed", 0),
            "mixed": labels.get("mixed", 0),
            "undetermined": labels.get("undetermined", 0),
            "not_applicable": labels.get("not_applicable", 0),
            "classifier_version": ok[0].language_report["classifier_version"] if ok else None,
        },
        "extraction_checks": {
            "entries": len(ok),
            "exact_copy": sum(c["exact_copy"] for c in checks),
            "normalized_copy": sum(c["normalized_copy"] for c in checks),
            "multi_sentence": sum(c["multi_sentence"] for c in checks),
            "label_leak": sum(c["label_leak"] for c in checks),
            "char_garble": sum(c["char_garble"] for c in checks),
            "any_corruption_signal": sum(1 for c in checks if c["corruption_signals"]),
            "note": "signals are a lower bound on corruption; the rate needs hand labels",
        },
    }


def _count(trace: list[dict[str, Any]], key) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in trace:
        k = key(t)
        if k is not None:
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


# --- trace rows ------------------------------------------------------------------


def _input_block(record: Tier1Record) -> dict[str, Any]:
    return {
        "utterance_a": record.utterance_a, "utterance_b": record.utterance_b,
        "utterance_a_id": record.utterance_a_id, "utterance_b_id": record.utterance_b_id,
        "input_sha256": pair_input_hash(record.eval_pair_id, record.language, record.utterance_a, record.utterance_b),
    }


def _skipped_row(group_id, config_id, record, status, error) -> dict[str, Any]:
    return {
        "schema": TRACE_SCHEMA, "group_id": group_id, "config": config_id, "status": status,
        "error": error, "eval_pair_id": record.eval_pair_id, "language": record.language,
        "input": _input_block(record),
    }


def _extraction_block(s: _Side, cfg) -> dict[str, Any]:
    block: dict[str, Any] = {
        "mode": cfg.extraction, "status": s.status, "error": s.error,
        "n_entries": len(s.entries) if s.entries is not None else 0,
        "prompt_id": None, "prompt_sha256": None, "raw_output": None, "model_distinction": None,
    }
    if s.native is not None:
        block.update(
            prompt_id=NATIVE_B_V1.prompt_id, prompt_sha256=NATIVE_B_V1.sha256,
            raw_output=s.native.raw_output,
        )
        if s.checks is not None:
            block["checks"] = s.checks
    if s.status == "extraction_failed":
        block["dropped_at_extraction"] = True
    if s.v4 is not None:
        block["model_distinction"] = dict(s.v4.distinction)
        block["v4_prompt_id"] = s.v4.prompt_id
        if cfg.extraction == "english_forced_prompt":
            block.update(prompt_id=s.v4.prompt_id, prompt_sha256=DEFAULT_EXTRACTION_PROMPT.sha256
                         if s.v4.prompt_id == DEFAULT_EXTRACTION_PROMPT.prompt_id else None,
                         raw_output=s.v4.raw_output)
    elif s.v4_error is not None:
        block["v4_error"] = str(s.v4_error)
        if cfg.extraction == "english_forced_prompt":
            block.update(prompt_id=DEFAULT_EXTRACTION_PROMPT.prompt_id,
                         prompt_sha256=DEFAULT_EXTRACTION_PROMPT.sha256, raw_output=s.v4_error.raw_output)
    return block


def _episode_row(group_id, config_id, record, status, sa: _Side, sb: _Side, cfg, lexicon) -> dict[str, Any]:
    dropped = [side.side for side in (sa, sb) if side.status == "extraction_failed"]
    return {
        "schema": TRACE_SCHEMA, "group_id": group_id, "config": config_id, "status": status,
        "eval_pair_id": record.eval_pair_id, "language": record.language,
        "input": _input_block(record),
        "extraction": {"a": _extraction_block(sa, cfg), "b": _extraction_block(sb, cfg)},
        "diagnosis": diagnose_dropped(
            lexicon=lexicon, lang=record.language, utterance_a=record.utterance_a,
            utterance_b=record.utterance_b, extraction_loss_stage=_STORAGE_LOSS_STAGE[config_id],
            sides_dropped=dropped,
        ),
    }


def _entry_id(utterance_id: str, index: int) -> str:
    return f"{utterance_id}_e{index}"


def _candidate_row(
    group_id, config_id, cfg, c: _Candidate, modes, lexicon, rows_by_mode, summary, *,
    threshold, tau, metric_name,
) -> dict[str, Any]:
    record = c.record
    entry_id_a, entry_id_b = _entry_id(c.a.utterance_id, c.ia), _entry_id(c.b.utterance_id, c.ib)
    row: dict[str, Any] = {
        "schema": TRACE_SCHEMA, "group_id": group_id, "config": config_id,
        "status": "ok" if c.proposed is not None else "judge_failed",
        "eval_pair_id": record.eval_pair_id, "language": record.language,
        "record_id": f"{entry_id_a}::{entry_id_b}",
        "input": _input_block(record),
        "extraction": {"a": _extraction_block(c.a, cfg), "b": _extraction_block(c.b, cfg)},
        "stored": {
            "a": _stored_block(entry_id_a, c.text_a, cfg, c.a, record.language),
            "b": _stored_block(entry_id_b, c.text_b, cfg, c.b, record.language),
        },
        "similarity": {
            "metric": metric_name, "score": c.sim, "text_a": c.text_a, "text_b": c.text_b,
            "decisive": cfg.merge_mechanism == "embedding_threshold",
        },
        "host": {
            "mechanism": cfg.merge_mechanism, "proposed": c.proposed,
            "tau": tau, "tau_used": cfg.merge_mechanism == "embedding_threshold",
            "judge": None,
        },
        "gate": None, "final": None, "diagnosis": None,
    }
    if c.judge is not None:
        row["host"]["judge"] = {
            "prompt_id": MERGE_JUDGE_V1.prompt_id, "prompt_sha256": MERGE_JUDGE_V1.sha256,
            "raw_output": c.judge.raw_output, "decision": c.judge.decision,
            "reason": c.judge.reason, "error": c.judge.error,
        }
    if c.proposed is None:
        return row

    row["diagnosis"] = diagnose(
        lexicon=lexicon, lang=record.language,
        utterance_a=record.utterance_a, utterance_b=record.utterance_b,
        stored_text_a=c.text_a, stored_text_b=c.text_b, host_proposed=c.proposed,
        storage_loss_stage=_STORAGE_LOSS_STAGE[config_id] or "storage",
        host_loss_stage=_HOST_LOSS_STAGE[config_id],
    )
    row["gate"], row["final"] = {}, {}
    for mode in modes:
        dist_a, dist_b = _mode_distinctions(mode, c, lexicon)
        if dist_a is None:  # lexicon+llm without a usable V4 extraction on a side
            row["gate"][mode] = {"applied": False, "skipped": "no V4 extraction for the model fallback"}
            row["final"][mode] = None
            continue
        if mode == "off":
            internal, compat = c.proposed, None
            row["gate"][mode] = {"applied": False, "proposed": c.proposed, "decision": c.proposed}
        else:
            g = apply_gate(c.proposed, dist_a, dist_b)
            internal, compat = g.decision, g.compatibility
            row["gate"][mode] = {
                "applied": True, "distinction_a": dist_a, "distinction_b": dist_b,
                "compatibility": g.compatibility, "proposed": g.proposed, "decision": g.decision,
                "vetoed": g.vetoed, "reason": g.reason,
            }
        decision = DECISION_TRANSLATION[internal]
        method = None if mode == "off" else mode
        entry_a = MemoryEntry(
            entry_id=entry_id_a, source_utterance_id=c.a.utterance_id,
            gold_entity_id=record.opaque_entity_id_a, language=record.language,
            gloss=c.text_a, surface=record.utterance_a,
            distinction=None if mode == "off" else translate_distinction(dist_a, extraction_method=method),
        )
        entry_b = MemoryEntry(
            entry_id=entry_id_b, source_utterance_id=c.b.utterance_id,
            gold_entity_id=record.opaque_entity_id_b, language=record.language,
            gloss=c.text_b, surface=record.utterance_b,
            distinction=None if mode == "off" else translate_distinction(dist_b, extraction_method=method),
        )
        out = PairwiseEvalRecord(
            record_id=row["record_id"], run_id=f"{group_id}-{mode_slug(mode)}",
            strategy=strategy_for(config_id, mode), entry_a=entry_a, entry_b=entry_b,
            decision=decision, similarity_score=c.sim, threshold=threshold, compatibility=compat,
            predicted_entity_id=entry_id_a if decision in ("merge", "supersede") else None,
        )
        rows_by_mode[mode].append(out)
        row["final"][mode] = {
            "decision": internal, "pairwise_decision": decision, "record_id": out.record_id,
        }
    return row


def _stored_block(entry_id: str, text: str, cfg, s: _Side, lang: str) -> dict[str, Any]:
    block = {"entry_id": entry_id, "text": text, "storage": cfg.storage}
    if cfg.config_id == "B":
        # language drift / script drift (classifier v2) and the automatic corruption signals
        # are separate fields; both were computed once, at extraction time
        block["output_language"] = s.language_report or output_language_report(s.utterance, text, lang)
        block["checks"] = s.checks
    return block


def _mode_distinctions(mode: str, c: _Candidate, lexicon: Lexicon):
    """``(distinction_a, distinction_b)`` the gate sees in ``mode``; ``(None, None)`` if unavailable."""
    if mode == "off":
        return {}, {}
    if mode == "lexicon":
        return dict(c.a.lex_dist), dict(c.b.lex_dist)
    if c.a.v4 is None or c.b.v4 is None:
        return None, None
    return (
        apply_lexicon(c.a.v4, lexicon, c.record.language).distinction,
        apply_lexicon(c.b.v4, lexicon, c.record.language).distinction,
    )


# --- output ----------------------------------------------------------------------


def write_outputs(out_dir: str | Path, result: StageAttributionResult) -> dict[str, str]:
    """Write ``run_config.json``, ``trace.jsonl``, ``summary.json`` and, per DK-Mem
    mode, ``<mode>/run_manifest.json`` + ``<mode>/pairwise_eval.jsonl``.

    Returns the paths written (as strings, keyed by file).
    """
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = result.run_config
    paths = {}
    p = out / "run_config.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    paths["run_config"] = str(p)
    p = out / "trace.jsonl"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for row in result.trace:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    paths["trace"] = str(p)
    p = out / "summary.json"
    p.write_text(json.dumps(result.summary, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    paths["summary"] = str(p)
    for mode, rows in result.rows_by_mode.items():
        d = out / mode_slug(mode)
        d.mkdir(exist_ok=True)
        write_run_manifest(
            d / "run_manifest.json", run_id=f"{cfg['group_id']}-{mode_slug(mode)}",
            strategy=strategy_for(cfg["pipeline_config"]["config_id"], mode), seed=cfg["seed"],
            backbone=cfg["backbone"], created_at=cfg["created_at"],
            pipeline_config=cfg["pipeline_config"]["config_id"], dkmem_mode=mode,
        )
        write_pairwise_eval(d / "pairwise_eval.jsonl", rows)
        paths[f"{mode}/pairwise_eval"] = str(d / "pairwise_eval.jsonl")
    return paths
