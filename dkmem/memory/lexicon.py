"""Data contract and loader for the DK-Mem distinction lexicon.

This defines the schema for ``distinction_features.json``: a static,
hand-curated table mapping source-language surface markers (kinship terms,
address pronouns, evidential suffixes, classifier/counter words, politeness
endings, deixis words, ...) to the ``Extraction.distinction`` key/value they
imply. It is the "lexicon" half of the lexicon-first, model-second cascade
described in DKMEM_NEW_RESEARCH_IDEA.md Sec 5.3: a deterministic dictionary
lookup meant to cover the common/closed-class cases cheaply, before any model
call is needed for the rest.

Scope of this module: parse and validate the JSON file into typed, queryable
objects (``load_lexicon`` -> ``Lexicon``). It does NOT match a lexicon
against an utterance -- scanning text for hits and building an
``Extraction.distinction`` dict is the DK-Mem extractor's job, built
separately on top of the ``Lexicon`` this module returns.

Current content and scope: Hindi kinship and register entries, matching the
in-scope classes in ``dkmem.memory.scope``. The loader itself accepts any
snake_case ``distinction_class``; keeping the file in scope is checked by
tests, not enforced here. ``name_variant`` is in scope for the project but has
no entries yet: a person's name is an open vocabulary, so whether and how a
dictionary covers it is undecided.

See ``dkmem/memory/distinction_features.md`` for the field-by-field format
and instructions for populating the JSON file.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dkmem.memory.extract import CLOSED_DISTINCTION_VALUES

__all__ = [
    "LEXICON_SCHEMA_VERSION",
    "MATCH_TYPES",
    "LexiconError",
    "LexiconEntry",
    "Lexicon",
    "load_lexicon",
]

LEXICON_SCHEMA_VERSION = "1.0"

# Matching mechanisms the (future) DK-Mem matcher must implement. This is a
# fixed set of *mechanisms*, not a linguistic taxonomy: adding one requires a
# code change to whatever matcher consumes a Lexicon, not just new data.
#   exact_word - the surface form equals a whole token
#   prefix     - a token starts with the surface form
#   suffix     - a token ends with the surface form (verb/case endings)
#   regex      - the surface form is a regular expression (compiled at load time)
MATCH_TYPES = frozenset({"exact_word", "prefix", "suffix", "regex"})

_ENTRY_REQUIRED_KEYS = frozenset({"id", "lang", "distinction_class", "match_type", "surface_forms", "value"})
_ENTRY_OPTIONAL_KEYS = frozenset({"priority", "notes"})
_DISTINCTION_CLASS_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LANG_RE = re.compile(r"^[a-z]{2,3}$")


class LexiconError(ValueError):
    """The lexicon file, or one of its entries, is malformed or conflicting."""


def _nonempty_str(obj: Any, name: str) -> None:
    value = getattr(obj, name)
    if not isinstance(value, str) or not value.strip():
        raise LexiconError(f"{type(obj).__name__}.{name} must be a non-empty string, got {value!r}")


@dataclass(frozen=True)
class LexiconEntry:
    """One surface-marker -> distinction value mapping.

    ``lang`` is a single lowercase 2-3 letter code (the language the marker
    belongs to, e.g. ``"hi"``, ``"tr"``). ``distinction_class`` is the
    ``Extraction.distinction`` key this entry fills (e.g. ``"kinship"``,
    ``"evidentiality"``); it is intentionally open-ended so DK-Mem can add
    classes the Mem0-style baseline prompts don't use (e.g.
    ``"spatial_deixis"``), but must look like a snake_case identifier.

    ``surface_forms`` are one or more literal markers (``exact_word``,
    ``prefix``, ``suffix``) or regular expressions (``regex``) that all imply
    the same ``value`` -- e.g. spelling/script variants of one word, or
    suffix allomorphs of one grammatical marker.

    ``value`` is the string written into ``Extraction.distinction[class]``.
    For the three classes the extraction pipeline already treats as
    closed-vocabulary (``politeness``, ``evidentiality``, ``temporal_deixis``;
    see ``dkmem.memory.extract.CLOSED_DISTINCTION_VALUES``), ``value`` is
    validated against that same fixed set so a lexicon entry can never
    produce a value the extraction schema would reject downstream.

    ``priority`` (default 0) is a hint for a future matcher to break ties
    when several entries match overlapping spans (e.g. prefer the entry with
    the higher priority, or the longest surface form); it has no effect in
    this module. ``notes`` is free-text documentation for the entry.
    """

    id: str
    lang: str
    distinction_class: str
    match_type: str
    surface_forms: tuple[str, ...]
    value: str
    priority: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        _nonempty_str(self, "id")

        _nonempty_str(self, "lang")
        if not _LANG_RE.match(self.lang):
            raise LexiconError(f"entry {self.id!r}: lang must be a lowercase 2-3 letter code, got {self.lang!r}")

        _nonempty_str(self, "distinction_class")
        if not _DISTINCTION_CLASS_RE.match(self.distinction_class):
            raise LexiconError(
                f"entry {self.id!r}: distinction_class must look like a snake_case identifier, "
                f"got {self.distinction_class!r}"
            )

        if self.match_type not in MATCH_TYPES:
            raise LexiconError(
                f"entry {self.id!r}: match_type must be one of {sorted(MATCH_TYPES)}, got {self.match_type!r}"
            )

        if not isinstance(self.surface_forms, tuple) or not self.surface_forms:
            raise LexiconError(f"entry {self.id!r}: surface_forms must be a non-empty list")
        for form in self.surface_forms:
            if not isinstance(form, str) or not form.strip():
                raise LexiconError(f"entry {self.id!r}: surface_forms entries must be non-empty strings, got {form!r}")
        if len(set(self.surface_forms)) != len(self.surface_forms):
            raise LexiconError(f"entry {self.id!r}: surface_forms has duplicates: {list(self.surface_forms)}")
        if self.match_type == "regex":
            for pattern in self.surface_forms:
                try:
                    re.compile(pattern)
                except re.error as e:
                    raise LexiconError(f"entry {self.id!r}: invalid regex {pattern!r}: {e}") from e

        _nonempty_str(self, "value")
        closed = CLOSED_DISTINCTION_VALUES.get(self.distinction_class)
        if closed is not None and self.value not in closed:
            raise LexiconError(
                f"entry {self.id!r}: value for closed class {self.distinction_class!r} "
                f"must be one of {sorted(closed)}, got {self.value!r}"
            )

        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise LexiconError(f"entry {self.id!r}: priority must be an int, got {type(self.priority).__name__}")

        if not isinstance(self.notes, str):
            raise LexiconError(f"entry {self.id!r}: notes must be a string, got {type(self.notes).__name__}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LexiconEntry":
        """Build one entry from its JSON object, rejecting unknown/missing keys."""
        if not isinstance(data, dict):
            raise LexiconError(f"entry must be an object, got {type(data).__name__}")
        keys = set(data)
        missing = _ENTRY_REQUIRED_KEYS - keys
        unknown = keys - _ENTRY_REQUIRED_KEYS - _ENTRY_OPTIONAL_KEYS
        if missing or unknown:
            raise LexiconError(f"entry {data.get('id', '<no id>')!r}: missing keys {sorted(missing)}, unexpected keys {sorted(unknown)}")
        surface_forms = data["surface_forms"]
        if not isinstance(surface_forms, list):
            raise LexiconError(f"entry {data.get('id')!r}: surface_forms must be a list, got {type(surface_forms).__name__}")
        try:
            return cls(
                id=data["id"],
                lang=data["lang"],
                distinction_class=data["distinction_class"],
                match_type=data["match_type"],
                surface_forms=tuple(surface_forms),
                value=data["value"],
                priority=data.get("priority", 0),
                notes=data.get("notes", ""),
            )
        except TypeError as e:
            raise LexiconError(f"entry {data.get('id')!r}: {e}") from e

    def to_dict(self) -> dict[str, Any]:
        """Plain, JSON-serializable dict (``surface_forms`` as a list)."""
        d = asdict(self)
        d["surface_forms"] = list(d["surface_forms"])
        return d


class Lexicon:
    """A validated, queryable set of ``LexiconEntry`` records.

    Read-only. Construct with ``load_lexicon``, not directly, so the
    cross-entry checks (unique ids, no conflicting surface forms) always run.
    """

    def __init__(self, entries: tuple[LexiconEntry, ...]) -> None:
        self.entries: tuple[LexiconEntry, ...] = tuple(entries)
        index: dict[tuple[str, str], list[LexiconEntry]] = {}
        for e in self.entries:
            index.setdefault((e.lang, e.distinction_class), []).append(e)
        self._by_lang_class = {k: tuple(v) for k, v in index.items()}

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def for_lang_class(self, lang: str, distinction_class: str) -> tuple[LexiconEntry, ...]:
        """Entries for exactly this ``(lang, distinction_class)`` pair, in file order."""
        return self._by_lang_class.get((lang, distinction_class), ())

    @property
    def langs(self) -> frozenset[str]:
        return frozenset(e.lang for e in self.entries)

    @property
    def distinction_classes(self) -> frozenset[str]:
        return frozenset(e.distinction_class for e in self.entries)


def load_lexicon(path: str | Path) -> Lexicon:
    """Load and fully validate a ``distinction_features.json`` file.

    Raises ``LexiconError`` (naming the file and, where possible, the entry)
    for: invalid JSON, an unsupported ``schema_version``, a malformed entry,
    a duplicate entry id, or two entries giving different values for the
    same ``(lang, distinction_class, match_type, surface_form)``. Identical
    duplicate mappings across entries are accepted.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        raise LexiconError(f"{path}: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LexiconError(f"{path}: invalid JSON: {e}") from e

    if not isinstance(data, dict) or "schema_version" not in data or "entries" not in data:
        raise LexiconError(f"{path}: top level must be an object with 'schema_version' and 'entries'")
    unknown = set(data) - {"schema_version", "description", "entries"}
    if unknown:
        raise LexiconError(f"{path}: unexpected top-level keys {sorted(unknown)}")
    if data["schema_version"] != LEXICON_SCHEMA_VERSION:
        raise LexiconError(
            f"{path}: unsupported schema_version {data['schema_version']!r}, expected {LEXICON_SCHEMA_VERSION!r}"
        )
    raw_entries = data["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise LexiconError(f"{path}: 'entries' must be a non-empty list")

    entries: list[LexiconEntry] = []
    seen_ids: set[str] = set()
    seen_surface: dict[tuple[str, str, str, str], str] = {}
    for i, raw in enumerate(raw_entries):
        try:
            entry = LexiconEntry.from_dict(raw)
        except LexiconError as e:
            raise LexiconError(f"{path}: entries[{i}]: {e}") from e

        if entry.id in seen_ids:
            raise LexiconError(f"{path}: entries[{i}]: duplicate id {entry.id!r}")
        seen_ids.add(entry.id)

        for form in entry.surface_forms:
            key = (entry.lang, entry.distinction_class, entry.match_type, form)
            prior = seen_surface.get(key)
            if prior is not None and prior != entry.value:
                raise LexiconError(
                    f"{path}: entries[{i}] ({entry.id!r}): surface form {form!r} for "
                    f"{entry.lang}/{entry.distinction_class}/{entry.match_type} already maps to "
                    f"{prior!r}, this entry says {entry.value!r}"
                )
            seen_surface[key] = entry.value

        entries.append(entry)

    return Lexicon(tuple(entries))
