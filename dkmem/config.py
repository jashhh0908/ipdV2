"""Experiment configuration vocabulary (DKMEM_NEW_RESEARCH_IDEA.md Sec 6.2, 6.3).

Names only: nothing here runs extraction, a merge judge, an embedding or a
baseline. It defines the labels a run carries so the stage-attribution
configurations A-D and DK-Mem ON/OFF are recorded the same way everywhere
(``run_manifest.json`` fields ``pipeline_config`` and ``dkmem_mode``, see
``dkmem.tier1.io.write_run_manifest``). The harness that runs A-D is
``dkmem.pipeline``.

Stage-attribution configurations (each is run with and without the DK-Mem
gate; the gate sits in front of the host's merge decision):

====== ======================= ================ =================== =======
config extraction              storage          merge decision      isolates
====== ======================= ================ =================== =======
A      English-forced prompt   English gloss    LLM judge           L1
B      native-language prompt  extractor output LLM judge           L2
C      none (verbatim)         surface text     LLM judge           L3
D      none (verbatim)         surface text     embedding threshold L4
====== ======================= ================ =================== =======

(L1 forced-English extraction, L2 extractor language drift, L3 merge-judge
collapse, L4 embedding collapse.) The "native-language prompt" of Config B is our
own frozen prompt ``native_b_v1`` (``dkmem.memory.native_extract.NATIVE_B_V1``): the
extractor is told to keep the user's language and script, and what it actually
wrote (language drift, script drift, corruption) is what B measures. It is **not**
the Mem0 baseline: a real Mem0 run, pinned to a version, is a separate
baseline row (Sec 6.3 of the plan) and does not exist in this repo yet. The LLM
judge is ``dkmem.memory.judge`` and the embedding metric
``dkmem.backends.embedding``.

DK-Mem mode: ``off`` (host only), ``lexicon`` (gate on, distinctions from the
lexicon alone) or ``lexicon+llm`` (gate on, lexicon first, model second).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "PIPELINE_CONFIG_IDS",
    "EXTRACTION_MODES",
    "STORAGE_MODES",
    "MERGE_MECHANISMS",
    "LOSS_STAGES",
    "PipelineConfig",
    "PIPELINE_CONFIGS",
    "get_pipeline_config",
    "DKMEM_MODES",
    "validate_dkmem_mode",
    "dkmem_gate_enabled",
    "STRATEGIES",
    "dkmem_mode_for_strategy",
]

PIPELINE_CONFIG_IDS = ("A", "B", "C", "D")

EXTRACTION_MODES = ("english_forced_prompt", "native_language_prompt", "none_verbatim")
STORAGE_MODES = ("english_gloss", "extractor_output", "surface_text")
MERGE_MECHANISMS = ("llm_judge", "embedding_threshold")
LOSS_STAGES = ("L1", "L2", "L3", "L4")


@dataclass(frozen=True)
class PipelineConfig:
    """One row of the stage-attribution table."""

    config_id: str
    extraction: str
    storage: str
    merge_mechanism: str
    isolates: str

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("config_id", self.config_id, PIPELINE_CONFIG_IDS),
            ("extraction", self.extraction, EXTRACTION_MODES),
            ("storage", self.storage, STORAGE_MODES),
            ("merge_mechanism", self.merge_mechanism, MERGE_MECHANISMS),
            ("isolates", self.isolates, LOSS_STAGES),
        ):
            if value not in allowed:
                raise ValueError(f"PipelineConfig.{name} must be one of {allowed}, got {value!r}")


PIPELINE_CONFIGS = MappingProxyType(
    {
        c.config_id: c
        for c in (
            PipelineConfig("A", "english_forced_prompt", "english_gloss", "llm_judge", "L1"),
            PipelineConfig("B", "native_language_prompt", "extractor_output", "llm_judge", "L2"),
            PipelineConfig("C", "none_verbatim", "surface_text", "llm_judge", "L3"),
            PipelineConfig("D", "none_verbatim", "surface_text", "embedding_threshold", "L4"),
        )
    }
)


def get_pipeline_config(config_id: str) -> PipelineConfig:
    """Look up a configuration by id ("A".."D")."""
    try:
        return PIPELINE_CONFIGS[config_id]
    except KeyError:
        raise KeyError(f"unknown pipeline config {config_id!r}; known: {sorted(PIPELINE_CONFIGS)}") from None


DKMEM_MODES = ("off", "lexicon", "lexicon+llm")


def validate_dkmem_mode(mode: str) -> str:
    """Return ``mode`` if it is one of ``DKMEM_MODES``, else raise ``ValueError``."""
    if mode not in DKMEM_MODES:
        raise ValueError(f"dkmem_mode must be one of {DKMEM_MODES}, got {mode!r}")
    return mode


def dkmem_gate_enabled(mode: str) -> bool:
    """True when the DK-Mem gate is applied (any mode but ``off``)."""
    return validate_dkmem_mode(mode) != "off"


# Strategy names allowed in run_manifest.json / pairwise_eval.jsonl: the
# baselines of Sec 6.3 plus the two DK-Mem variants. Must equal the ``strategy``
# enum in both JSON schemas (checked by tests). Mem0 under config A vs B, and
# store-surface-only / embedding-threshold as configs C / D, are told apart by
# ``pipeline_config``, not by separate strategy names.
STRATEGIES = (
    "mem0",
    "a-mem",
    "store-surface-only",
    "embedding-threshold",
    "raise-tau",
    "prompt-informed-judge",
    "flat-dense-rag",
    "dk-mem-lexicon",
    "dk-mem-lexicon-llm",
)

_STRATEGY_DKMEM_MODE = {"dk-mem-lexicon": "lexicon", "dk-mem-lexicon-llm": "lexicon+llm"}


def dkmem_mode_for_strategy(strategy: str) -> str:
    """The DK-Mem mode a strategy implies: its own variant for the two DK-Mem
    strategies, ``off`` for every other one."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    return _STRATEGY_DKMEM_MODE.get(strategy, "off")
