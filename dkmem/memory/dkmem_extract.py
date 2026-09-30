"""DK-Mem extraction: the lexicon-first, model-second cascade.

Combines the deterministic ``dkmem.memory.matcher`` with the existing,
**unmodified** Mem0/Qwen baseline extractor (``dkmem.memory.extract``), per
research_idea_context.md Sec 4(a):

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
       - the lexicon found exactly one candidate value -> use it, overriding
         the model's own guess (lexicon-first: a confident deterministic
         match is trusted over the model, matching the research design's
         "cheap deterministic primitives beating LLM reasoning" premise);
       - the lexicon found more than one *different* candidate value for
         the class (a real conflict on this utterance, not the same value
         confirmed twice) -> this is the "ambiguous" case in
         research_idea_context.md Sec 4(a) ("a small LLM call only for
         spans the lexicon flags as ambiguous"): the lexicon's own
         tie-broken guess (from ``match_distinctions``) is set aside and
         the model's value is kept instead;
       - the lexicon found nothing for the class ("unmatched") -> the
         model's value is kept unchanged (if it has none either, the class
         is simply absent from the result, as before).

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
from dkmem.memory.prompts import MEM0_EXTRACTION_V1, PromptTemplate
from dkmem.memory.schema import Extraction, ProbeItem

__all__ = [
    "DKMEM_LEXICON_SUFFIX",
    "dkmem_prompt_id",
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


def apply_lexicon(extraction: Extraction, lexicon: Lexicon, lang: str) -> Extraction:
    """Return a copy of ``extraction`` with ``distinction`` lexicon-augmented.

    Scans ``extraction.surface`` (equal to the original utterance, per
    ``dkmem.memory.extract``'s own validation) against ``lexicon``, once per
    ``-``-separated component of ``lang`` (so a code-mixed tag like
    ``"hi-en"`` checks both ``"hi"`` and ``"en"`` entries). See the module
    docstring for exactly how a confident/ambiguous/unmatched lexicon result
    is combined with the model's own ``distinction``.

    Everything except ``distinction`` and ``prompt_id`` is copied verbatim.
    """
    components = _parse_lang_tag(lang)
    utterance = extraction.surface

    resolved: dict[str, str] = {}
    for component in components:
        resolved.update(match_distinctions(lexicon, utterance, component))
    ambiguous = _ambiguous_classes(lexicon, utterance, components)

    merged = dict(extraction.distinction)  # model's own values: the fallback
    for distinction_class, value in resolved.items():
        if distinction_class not in ambiguous:
            merged[distinction_class] = value
        # else: ambiguous lexicon evidence for this class on this utterance;
        # keep whichever value (if any) the model already contributed.

    return replace(extraction, distinction=merged, prompt_id=dkmem_prompt_id(extraction.prompt_id))


def dkmem_extract_many(
    items: Sequence[tuple[ProbeItem, str]],
    generator: TextGenerator,
    lexicon: Lexicon,
    cache: ExtractionCache | None = None,
    params: GenerationParams | None = None,
    *,
    prompt: PromptTemplate = MEM0_EXTRACTION_V1,
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
    prompt: PromptTemplate = MEM0_EXTRACTION_V1,
    backend_info: Mapping[str, Any] | None = None,
) -> Extraction:
    """Single-item ``dkmem_extract_many``."""
    return dkmem_extract_many(
        [(probe, side)], generator, lexicon, cache, params, prompt=prompt, backend_info=backend_info
    )[0]
