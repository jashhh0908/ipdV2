"""Deterministic lexicon matcher: scans an utterance for ``LexiconEntry`` hits.

Given a loaded ``Lexicon`` (see ``dkmem.memory.lexicon``) and one utterance,
``match_distinctions`` returns a ``dict[str, str]`` in exactly the shape
``Extraction.distinction`` expects: one value per distinction class the
lexicon recognizes in the utterance.

This module implements only the deterministic *mechanism* for scanning text
against ``exact_word``/``prefix``/``suffix``/``regex`` entries and resolving
conflicts by priority (see "Tie-break rule" below). It does not do
sentence-level or contextual disambiguation -- e.g. it cannot use verb tense
to pick a ``temporal_deixis`` reading, and a marker that needs that is simply
either matched context-free or not matched at all. It does not construct
``Extraction`` records and does not import anything from
``dkmem.memory.extract`` (the Mem0-style baseline prompts/extractor): the
two stay fully independent, so the lexicon path can be run and evaluated
(lexicon-only vs lexicon+LLM, per research_idea_context.md Sec 4a) without
either depending on the other.

Tokenization
------------
An utterance is split on whitespace; each token then has leading/trailing
punctuation/symbol characters stripped (Unicode general category ``P*``/
``S*``), so sentence-final punctuation across scripts (".", ",", Hindi "।",
Japanese "。", a quote mark or parenthesis at a token's edge, ...) doesn't
block a match. This is deliberately *not* Python's ``\\W`` (non-word
character): ``\\W`` also excludes combining marks, which would wrongly
trim the end off many real words -- most Devanagari words ending in a
dependent vowel sign (a matra, Unicode category ``Mn``/``Mc``, e.g. the
"ी" in "चाची") end in something ``\\W`` does not consider a word
character, even with no actual punctuation present. Trimming only
punctuation/symbol categories keeps such marks as part of the token. A
character in the middle of a token (e.g. the apostrophe in "Ankara'ya") is
never touched, regardless of its category, since only the edges are
scanned inward.

This is a simple, script-agnostic tokenizer, not a linguistic word
segmenter: a language that doesn't use whitespace between words (Japanese,
Chinese) is not segmented into words, so all four match types effectively
operate on whatever whitespace-delimited chunk contains the marker for such
text.

``exact_word``/``prefix``/``suffix`` compare the token and the entry's
surface form case-insensitively (Unicode ``casefold``). ``regex`` patterns
are applied to the token as-is via ``re.search`` (no implicit case-folding);
write ``(?i)`` into the pattern if case-insensitivity is wanted. Anchors
(``^``/``$``) refer to the token's boundaries, not the whole utterance.

Tie-break rule
--------------
Multiple entries can validly match one utterance -- e.g. two different
tokens each carrying a different distinction feature (fine: each becomes
its own key in the result). A genuine conflict is two matches for the
*same* ``distinction_class`` with *different* values, since
``Extraction.distinction`` holds one value per class. Conflicts are
resolved deterministically, per class, in this order:

1. Higher ``LexiconEntry.priority`` wins.
2. Still tied: the match at the earlier token position wins. This is an
   arbitrary-but-fixed rule for determinism, not a linguistic claim that
   earlier markers matter more -- give an entry a higher ``priority`` if it
   should win regardless of position.
3. Still tied: the longer matched span wins (the literal matched text for
   ``exact_word``/``prefix``/``suffix``; the regex match's ``group(0)`` for
   ``regex``) -- prefers the more specific marker.
4. Still tied: the lexicographically lower ``LexiconEntry.id`` wins, purely
   so the result never depends on entry order in the JSON file or on dict
   iteration order.

Two matches that agree on the same value for a class are not a conflict:
that value is used once, regardless of how many entries/tokens produced it.

Known limitation: step 3's span slicing for ``prefix``/``suffix`` assumes
``str.casefold()`` does not change a string's length, true for every script
in the current lexicon (Latin ASCII, Devanagari, Turkish). A surface form
that changes length under casefolding (rare, e.g. German "ß") could shift
the reported ``Match.span`` slightly; it does not affect whether a match
occurs.

Suffix stem-length guard
-------------------------
A bare ``endswith`` check for ``suffix`` entries would also match a token
that *is* the suffix on its own (zero-length stem) or leaves only one
character before it -- never a real instance of a word bearing that suffix.
This matters concretely for the lexicon's Turkish evidential suffix entry
(``-mış``/``-miş``/``-muş``/``-müş``): the shortest attested Turkish verb
roots are two letters (e.g. "de-" -> "demiş", "ol-" -> "olmuş"), so a token
with fewer than ``_MIN_SUFFIX_STEM_LENGTH`` (2) characters before the suffix
cannot be that suffix attached to a genuine verb root and is rejected.

This is a narrow, mechanical guard, not morphological analysis: it does not
verify the remaining stem is an actual verb root (that would need a root
lexicon), and it cannot distinguish the finite-verb evidential/reportative
use of ``-mIş`` from its adjectival/nominalized participle use, which is
identical on the surface and NOT reported speech (e.g. "pişmiş yemek",
"cooked food", uses the same suffix on the same kind of stem length without
implying hearsay). Resolving that would require syntactic context this
module deliberately does not use -- see "Out of scope" in the package
docstring and the caveat repeated in
``dkmem/memory/distinction_features.md``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from dkmem.memory.lexicon import Lexicon, LexiconEntry

__all__ = ["Match", "find_matches", "match_distinctions"]

# See "Suffix stem-length guard" above.
_MIN_SUFFIX_STEM_LENGTH = 2


def _is_trimmable(ch: str) -> bool:
    """True for a punctuation or symbol character (Unicode category ``P*``/
    ``S*``). Deliberately narrower than "not a word character" (``\\W``):
    that would also match combining marks such as Devanagari vowel signs,
    which are part of the word, not decoration around it.
    """
    return unicodedata.category(ch)[0] in ("P", "S")


def _tokenize(utterance: str) -> list[str]:
    """Whitespace-split ``utterance``, trimming punctuation/symbol characters
    off each token's ends (see ``_is_trimmable``). All-punctuation fragments
    (which trim to empty) are dropped.
    """
    tokens = []
    for raw in utterance.split():
        start, end = 0, len(raw)
        while start < end and _is_trimmable(raw[start]):
            start += 1
        while end > start and _is_trimmable(raw[end - 1]):
            end -= 1
        token = raw[start:end]
        if token:
            tokens.append(token)
    return tokens


def _matched_span(entry: LexiconEntry, token: str) -> str | None:
    """The literal substring of ``token`` that ``entry`` matches, or ``None``."""
    if entry.match_type == "regex":
        for pattern in entry.surface_forms:
            m = re.search(pattern, token)
            if m:
                return m.group(0)
        return None

    token_cf = token.casefold()
    for form in entry.surface_forms:
        form_cf = form.casefold()
        if entry.match_type == "exact_word":
            if token_cf == form_cf:
                return token
        elif entry.match_type == "prefix":
            if token_cf.startswith(form_cf):
                return token[: len(form)]
        elif entry.match_type == "suffix":
            if token_cf.endswith(form_cf) and len(token) - len(form) >= _MIN_SUFFIX_STEM_LENGTH:
                return token[len(token) - len(form) :]
        else:  # pragma: no cover - LexiconEntry.__post_init__ already rejects this
            raise AssertionError(f"unhandled match_type {entry.match_type!r}")
    return None


@dataclass(frozen=True)
class Match:
    """One lexicon hit: ``entry`` matched ``span`` in the token at ``token_index``."""

    entry: LexiconEntry
    token_index: int
    token: str
    span: str


def find_matches(lexicon: Lexicon, utterance: str, lang: str) -> tuple[Match, ...]:
    """Every hit of a ``lang``-tagged entry in ``lexicon`` against ``utterance``.

    ``lang`` is a single lexicon language code (e.g. ``"hi"``), matched
    exactly against ``LexiconEntry.lang``; it is not parsed or split, so for
    a code-mixed utterance, call this once per candidate language and merge.
    An unknown ``lang`` (no entries for it) is not an error: it simply
    yields no hits.

    Returns hits in token order (ties in the lexicon's entry order); does
    not resolve conflicts between them -- see ``match_distinctions``.
    """
    tokens = _tokenize(utterance)
    hits: list[Match] = []
    for idx, token in enumerate(tokens):
        for entry in lexicon:
            if entry.lang != lang:
                continue
            span = _matched_span(entry, token)
            if span is not None:
                hits.append(Match(entry=entry, token_index=idx, token=token, span=span))
    return tuple(hits)


def _tie_break_key(match: Match) -> tuple[int, int, int, str]:
    # Ascending sort; priority and span length are negated so higher/longer sort first.
    return (-match.entry.priority, match.token_index, -len(match.span), match.entry.id)


def match_distinctions(lexicon: Lexicon, utterance: str, lang: str) -> dict[str, str]:
    """Scan ``utterance`` and return a distinction dict shaped for ``Extraction``.

    One key per distinction class with at least one match. A class with
    conflicting matched values is resolved by the tie-break rule documented
    at the top of this module. An utterance with no matches returns ``{}``.
    """
    hits = find_matches(lexicon, utterance, lang)
    by_class: dict[str, list[Match]] = {}
    for hit in hits:
        by_class.setdefault(hit.entry.distinction_class, []).append(hit)

    result: dict[str, str] = {}
    for distinction_class, class_hits in by_class.items():
        values = {h.entry.value for h in class_hits}
        if len(values) == 1:
            result[distinction_class] = next(iter(values))
        else:
            winner = min(class_hits, key=_tie_break_key)
            result[distinction_class] = winner.entry.value
    return result
