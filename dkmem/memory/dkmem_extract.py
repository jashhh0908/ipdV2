"""DK-Mem extraction: the lexicon-first, model-second cascade.

Combines the deterministic ``dkmem.memory.matcher`` with the existing,
**unmodified** Mem0/Qwen baseline extractor (``dkmem.memory.extract``), per
DKMEM_NEW_RESEARCH_IDEA.md Sec 5.3:

    1. Run the frozen baseline extraction (``extract``/``extract_many``, or
       their ``cached_extract``/``cached_extract_many`` wrappers) exactly as
       before: same prompt, same ``GenerationParams`` handling, same
       ``ExtractionCache``. This yields ``gloss``, ``surface``,
       ``lang_profile``, and the model's own guess at ``distinction``.
    2. Scan the utterance with ``dkmem.memory.matcher.match_distinctions``,
       against a caller-supplied ``Lexicon`` (``dkmem.memory.lexicon`` --
       **not** modified or extended here; another team member owns
       ``distinction_features.json``'s real content).
    3. For each distinction class:
       - the lexicon found exactly one candidate value -> use it
         (lexicon-first: a confident deterministic match is trusted over
         the model, matching the research design's "cheap deterministic
         primitives beating LLM reasoning" premise);
       - the lexicon found more than one *different* candidate value for
         the class (a real conflict on this utterance, not the same value
         confirmed twice) -> this is the "ambiguous" case in
         DKMEM_NEW_RESEARCH_IDEA.md Sec 5.3 and research_idea_context.md
         ("a small LLM call only for spans the lexicon flags as ambiguous"):
         the lexicon's own tie-broken guess (from ``match_distinctions``)
         is set aside and the model's value for that class is used instead
         (if the model has none, the class is absent);
       - the lexicon found nothing for the class ("unmatched") -> the class
         is absent. The model is NOT consulted: Sec 5.3 gives it no role
         outside lexicon-flagged ambiguity, so a term the lexicon does not
         cover is not tagged even if the model would tag it. (Until the
         spec was applied literally on 2026-10-08 the model's value was
         kept for unmatched classes too; results produced by that earlier
         cascade are not comparable.)
    Consequently the model is needed for an utterance only when
    ``ambiguous_distinction_classes`` is non-empty, and ``resolve_distinction``
    takes the extraction as optional.

The result is a new ``Extraction`` with the same ``pair_id``/``side``/
``gloss``/``surface``/``lang_profile``/``backbone``/``seed`` as the baseline
call, ``distinction`` replaced by the merge above, and ``prompt_id``
suffixed (see ``dkmem_prompt_id``) to record that its ``distinction`` is
lexicon-augmented -- so it is never mistaken for, and never collides in the
``ExtractionCache`` with, a pure baseline record.

Caching: only the baseline call is cached, via the existing
``ExtractionCache``, exactly as it already is -- this module does not read
or write it directly. The lexicon-merge step is a cheap, pure function of an
already-produced ``Extraction``, and persisting it through the *same* cache
would not be safe: ``ExtractionCache`` verifies that a cached record's
fields are exactly what ``dkmem.memory.extract`` reconstructs by re-parsing
``raw_output``, an invariant a lexicon-overridden ``distinction``
deliberately breaks.

Out of scope here, as elsewhere in this pipeline so far: consolidation/merge
gating between different utterances, and sentence-level/contextual
disambiguation within one utterance (e.g. using verb tense to pick a
``temporal_deixis`` reading) -- this module only decides, per distinction
class and per utterance, whether to trust the lexicon or the model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from dkmem.backends.llm import GenerationParams
from dkmem.memory.cache import ExtractionCache, cached_extract_many
from dkmem.memory.extract import TextGenerator, _parse_lang_tag, extract_many
from dkmem.memory.lexicon import Lexicon
from dkmem.memory.matcher import find_matches, match_distinctions
from dkmem.memory.prompts import DEFAULT_EXTRACTION_PROMPT, PromptTemplate
from dkmem.memory.schema import Extraction, ProbeItem

__all__ = [
    "DKMEM_LEXICON_SUFFIX",
    "dkmem_prompt_id",
    "ambiguous_distinction_classes",
    "resolve_distinction",
    "apply_lexicon",
    "dkmem_extract_many",
    "dkmem_extract",
]

DKMEM_LEXICON_SUFFIX = "+dkmem_lexicon"


def dkmem_prompt_id(base_prompt_id: str) -> str:
    """The ``prompt_id`` recorded for a lexicon-augmented ``Extraction``.

    Distinct from ``base_prompt_id`` so a DK-Mem result can never be
    mistaken for, or share a cache key with, the untouched Mem0 baseline
    record it was built from.
    """
    return base_prompt_id + DKMEM_LEXICON_SUFFIX


def _ambiguous_classes(lexicon: Lexicon, utterance: str, components: Sequence[str]) -> set[str]:
    """Distinction classes where the lexicon's own hits disagree on the
    value (more than one distinct candidate value), across all ``components``.

    ``match_distinctions`` always resolves such a conflict to one value via
    its documented priority tie-break; this function exists only to decide
    whether *this integration layer* should trust that resolved value, or
    set it aside in favor of the model's own answer (see module docstring).
    """
    by_class: dict[str, set[str]] = {}
    for component in components:
        for hit in find_matches(lexicon, utterance, component):
            by_class.setdefault(hit.entry.distinction_class, set()).add(hit.entry.value)
    return {cls for cls, values in by_class.items() if len(values) > 1}


def ambiguous_distinction_classes(lexicon: Lexicon, utterance: str, lang: str) -> set[str]:
    """The distinction classes the lexicon flags as ambiguous on ``utterance`` (more than one distinct
    candidate value, across the components of ``lang``): the only classes for which the lexicon+LLM
    cascade consults the model (Sec 5.3)."""
    return _ambiguous_classes(lexicon, utterance, _parse_lang_tag(lang))


def resolve_distinction(
    lexicon: Lexicon, utterance: str, lang: str, extraction: Extraction | None
) -> dict[str, str] | None:
    """The lexicon+LLM ``distinction`` of one utterance (see the module docstring).

    Lexicon-resolved values for every unambiguous class; for each class the lexicon flags as ambiguous, the
    model's value from ``extraction`` when it has one. Returns ``None`` when the utterance has an ambiguous
    class but ``extraction`` is ``None`` (the model's answer is needed and unavailable): the caller must not
    guess. With no ambiguity the extraction is not needed at all and may be ``None``.
    """
    components = _parse_lang_tag(lang)
    resolved: dict[str, str] = {}
    for component in components:
        resolved.update(match_distinctions(lexicon, utterance, component))
    ambiguous = _ambiguous_classes(lexicon, utterance, components)
    result = {cls: value for cls, value in resolved.items() if cls not in ambiguous}
    if not ambiguous:
        return result
    if extraction is None:
        return None
    for cls in sorted(ambiguous):
        if cls in extraction.distinction:
            result[cls] = extraction.distinction[cls]
    return result


def apply_lexicon(extraction: Extraction, lexicon: Lexicon, lang: str) -> Extraction:
    """Return a copy of ``extraction`` with ``distinction`` replaced by ``resolve_distinction``.

    Scans ``extraction.surface`` (equal to the original utterance, per
    ``dkmem.memory.extract``'s own validation) against ``lexicon``, once per
    ``-``-separated component of ``lang`` (so a code-mixed tag like
    ``"hi-en"`` checks both ``"hi"`` and ``"en"`` entries). The model's own
    ``distinction`` is used only for classes the lexicon flags as ambiguous.

    Everything except ``distinction`` and ``prompt_id`` is copied verbatim.
    """
    distinction = resolve_distinction(lexicon, extraction.surface, lang, extraction)
    return replace(extraction, distinction=distinction, prompt_id=dkmem_prompt_id(extraction.prompt_id))


def dkmem_extract_many(
    items: Sequence[tuple[ProbeItem, str]],
    generator: TextGenerator,
    lexicon: Lexicon,
    cache: ExtractionCache | None = None,
    params: GenerationParams | None = None,
    *,
    prompt: PromptTemplate = DEFAULT_EXTRACTION_PROMPT,
    backend_info: Mapping[str, Any] | None = None,
) -> list[Extraction]:
    """Baseline-extract ``items`` (cached if ``cache`` is given, else not),
    then lexicon-augment each result, in order.

    Batches the baseline call exactly as ``extract_many``/
    ``cached_extract_many`` already do. If the baseline call raises
    ``ExtractionBatchError`` (or ``ExtractionError``), it propagates
    unmodified before any lexicon augmentation runs: the lexicon step never
    sees an item that failed baseline extraction, since there is nothing to
    augment.
    """
    if cache is not None:
        baseline = cached_extract_many(
            items, generator, cache, params, prompt=prompt, backend_info=backend_info
        )
    else:
        baseline = extract_many(items, generator, params, prompt=prompt)
    return [
        apply_lexicon(extraction, lexicon, probe.lang)
        for (probe, _side), extraction in zip(items, baseline)
    ]


def dkmem_extract(
    probe: ProbeItem,
    side: str,
    generator: TextGenerator,
    lexicon: Lexicon,
    cache: ExtractionCache | None = None,
    params: GenerationParams | None = None,
    *,
    prompt: PromptTemplate = DEFAULT_EXTRACTION_PROMPT,
    backend_info: Mapping[str, Any] | None = None,
) -> Extraction:
    """Single-item ``dkmem_extract_many``."""
    return dkmem_extract_many(
        [(probe, side)], generator, lexicon, cache, params, prompt=prompt, backend_info=backend_info
    )[0]
