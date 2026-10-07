"""No-LLM extraction for the ``dk-mem-lexicon`` Tier 1 strategy.

``dkmem.memory.dkmem_extract`` always calls a generator (it overlays the
lexicon *on top of* a Mem0-style baseline call -- "model-second" is still a
model call for whatever the lexicon leaves unmatched). Team B's
``dk-mem-lexicon`` strategy is the other half of the paper's own ablation:
"Extra LLM calls: zero in the lexicon-only variant" (DKMEM_NEW_RESEARCH_IDEA.md
Sec 5.6) and ``run_manifest.backbone: null``. Nothing existing produces that,
so this module adds it, without touching ``dkmem_extract.py`` or any other
core module.

Honesty about ``gloss``: ``Extraction.gloss`` is normally "the normalized
English retrieval key" produced by the baseline LLM. With no LLM call at
all, there is no translation step, so ``gloss`` here is the raw surface
utterance, unchanged -- this is recorded plainly, not disguised as a real
English gloss.
"""

from __future__ import annotations

from dkmem.memory.extract import _parse_lang_tag, derive_lang_profile
from dkmem.memory.lexicon import Lexicon
from dkmem.memory.matcher import match_distinctions
from dkmem.memory.schema import SIDES, Extraction

__all__ = ["NO_BACKBONE", "LEXICON_ONLY_PROMPT_ID", "extract_lexicon_only"]

# Extraction.backbone/prompt_id require non-empty strings (schema.py is not
# modified for this), so "no LLM ran" is recorded as an explicit sentinel
# rather than an empty/null value. dkmem.tier1.io translates this sentinel
# to JSON null in run_manifest.json's backbone field -- the one place the
# external contract actually allows null.
NO_BACKBONE = "none-lexicon-only"
LEXICON_ONLY_PROMPT_ID = "lexicon_only_no_llm_v1"


def extract_lexicon_only(
    pair_id: str, side: str, utterance: str, lang: str, lexicon: Lexicon, *, seed: int
) -> Extraction:
    """Build an ``Extraction`` for one utterance side without any model call.

    ``distinction`` comes entirely from ``dkmem.memory.matcher.
    match_distinctions``, once per ``-``-separated component of ``lang``
    (same convention as ``dkmem_extract.apply_lexicon``). ``gloss`` and
    ``surface`` are both the raw utterance -- see the module docstring.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")

    distinction: dict[str, str] = {}
    for component in _parse_lang_tag(lang):
        distinction.update(match_distinctions(lexicon, utterance, component))

    return Extraction(
        pair_id=pair_id,
        side=side,
        raw_output="",
        gloss=utterance,
        distinction=distinction,
        surface=utterance,
        lang_profile=derive_lang_profile(lang, utterance),
        backbone=NO_BACKBONE,
        prompt_id=LEXICON_ONLY_PROMPT_ID,
        seed=seed,
    )
