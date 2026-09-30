"""Frozen data contracts for DK-Mem experiments.

Three record types flow through the pipeline and are logged as JSONL:

- ``ProbeItem``: a minimal-pair conflation probe (two utterances + gold label).
- ``Extraction``: the write-side dual-field extraction of one side of a probe.
- ``MergeEvent``: one consolidation decision taken by a merge policy on a pair.

These are plain stdlib dataclasses with type/required-field validation in
``__post_init__``. No extraction, memory, retrieval, or merge logic lives here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Iterator, TypeVar

__all__ = [
    "SIDES",
    "DECISIONS",
    "SchemaError",
    "ProbeItem",
    "Extraction",
    "MergeEvent",
    "write_jsonl",
    "read_jsonl",
]

SIDES = ("a", "b")
DECISIONS = ("merge", "supersede", "keep_both", "link_unresolved")

T = TypeVar("T", bound="_Record")


class SchemaError(ValueError):
    """Raised when a record violates its data contract."""


# --- validation helpers -----------------------------------------------------


def _check_str(obj: Any, name: str, *, non_empty: bool = False) -> None:
    value = getattr(obj, name)
    if not isinstance(value, str):
        raise SchemaError(f"{type(obj).__name__}.{name} must be str, got {type(value).__name__}")
    if non_empty and not value.strip():
        raise SchemaError(f"{type(obj).__name__}.{name} must be a non-empty string")


def _check_optional_str(obj: Any, name: str) -> None:
    if getattr(obj, name) is not None:
        _check_str(obj, name)


def _check_bool(obj: Any, name: str) -> None:
    value = getattr(obj, name)
    if not isinstance(value, bool):
        raise SchemaError(f"{type(obj).__name__}.{name} must be bool, got {type(value).__name__}")


def _check_int(obj: Any, name: str) -> None:
    value = getattr(obj, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{type(obj).__name__}.{name} must be int, got {type(value).__name__}")


def _is_finite_number(value: Any) -> bool:
    """True for int/float (not bool) that are finite, i.e. not NaN or +/-inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # int too large to convert to float
        return False


def _coerce_float(obj: Any, name: str) -> None:
    value = getattr(obj, name)
    if not _is_finite_number(value):
        raise SchemaError(f"{type(obj).__name__}.{name} must be a finite number, got {value!r}")
    object.__setattr__(obj, name, float(value))


def _check_choice(obj: Any, name: str, choices: tuple[str, ...]) -> None:
    value = getattr(obj, name)
    if value not in choices:
        raise SchemaError(f"{type(obj).__name__}.{name} must be one of {choices}, got {value!r}")


def _copy_mapping(obj: Any, name: str, value_ok, value_desc: str) -> None:
    """Validate a str-keyed mapping and store a defensive copy."""
    value = getattr(obj, name)
    if not isinstance(value, dict):
        raise SchemaError(f"{type(obj).__name__}.{name} must be dict, got {type(value).__name__}")
    for k, v in value.items():
        if not isinstance(k, str) or not value_ok(v):
            raise SchemaError(
                f"{type(obj).__name__}.{name} must map str -> {value_desc}, got {k!r}: {v!r}"
            )
    object.__setattr__(obj, name, dict(value))


# --- base -------------------------------------------------------------------


class _Record:
    """Shared dict/JSON (de)serialization for the schema dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable dict of this record."""
        return asdict(self)  # type: ignore[call-overload]

    def to_json(self) -> str:
        """Return this record as a single JSON line (no trailing newline)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Build a record from a dict, rejecting unknown and missing keys."""
        if not isinstance(data, dict):
            raise SchemaError(f"{cls.__name__} expects a dict, got {type(data).__name__}")
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = set(data) - names
        if unknown:
            raise SchemaError(f"{cls.__name__} got unknown fields: {sorted(unknown)}")
        try:
            return cls(**data)
        except TypeError as e:  # missing required fields
            raise SchemaError(f"{cls.__name__}: {e}") from e

    @classmethod
    def from_json(cls: type[T], line: str) -> T:
        """Parse one JSON line into a record."""
        return cls.from_dict(json.loads(line))


# --- contracts --------------------------------------------------------------


@dataclass(frozen=True)
class ProbeItem(_Record):
    """A minimal-pair conflation probe.

    Two utterances that differ in language ``lang`` only by a distinction of
    class ``distinction_class`` (e.g. kinship), and may be identical after
    English normalization. ``gold_same_entity`` is the human gold label.
    A ``distinction_value_*`` of ``None`` means that side is unmarked for the
    feature (the underdetermined case).
    """

    pair_id: str
    utt_a: str
    utt_b: str
    lang: str
    distinction_class: str
    distinction_value_a: str | None
    distinction_value_b: str | None
    gold_same_entity: bool
    retrieval_query: str
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("pair_id", "utt_a", "utt_b", "lang", "distinction_class"):
            _check_str(self, name, non_empty=True)
        _check_optional_str(self, "distinction_value_a")
        _check_optional_str(self, "distinction_value_b")
        _check_bool(self, "gold_same_entity")
        _check_str(self, "retrieval_query")
        _check_str(self, "notes")


@dataclass(frozen=True)
class Extraction(_Record):
    """Write-side dual-field extraction for one side (``"a"``/``"b"``) of a probe.

    ``gloss`` is the normalized English retrieval key, ``distinction`` the
    source-language discriminative features (e.g. ``{"kin": "chachi"}``;
    empty = unmarked), ``surface`` the verbatim original, and ``lang_profile``
    the language mix (e.g. ``{"hi": 0.7, "en": 0.3, "script_mix": 0.4}``).
    ``raw_output`` is the unparsed extractor output, kept for auditing.
    """

    pair_id: str
    side: str
    raw_output: str
    gloss: str
    distinction: dict[str, str]
    surface: str
    lang_profile: dict[str, float]
    backbone: str
    prompt_id: str
    seed: int

    def __post_init__(self) -> None:
        _check_str(self, "pair_id", non_empty=True)
        _check_choice(self, "side", SIDES)
        for name in ("raw_output", "gloss", "surface"):
            _check_str(self, name)
        _copy_mapping(self, "distinction", lambda v: isinstance(v, str), "str")
        _copy_mapping(self, "lang_profile", _is_finite_number, "finite number")
        object.__setattr__(
            self, "lang_profile", {k: float(v) for k, v in self.lang_profile.items()}
        )
        _check_str(self, "backbone", non_empty=True)
        _check_str(self, "prompt_id", non_empty=True)
        _check_int(self, "seed")


@dataclass(frozen=True)
class MergeEvent(_Record):
    """One consolidation decision made by merge ``policy`` on a probe pair.

    ``sim`` is the similarity score compared against threshold ``tau``; which
    representation it is computed on (gloss, surface, ...) depends on the
    policy. ``decision`` is one of ``DECISIONS`` and ``reason`` a short
    explanation.
    """

    pair_id: str
    policy: str
    tau: float
    sim: float
    decision: str
    reason: str
    backbone: str
    seed: int
    prompt_id: str

    def __post_init__(self) -> None:
        _check_str(self, "pair_id", non_empty=True)
        _check_str(self, "policy", non_empty=True)
        _coerce_float(self, "tau")
        _coerce_float(self, "sim")
        _check_choice(self, "decision", DECISIONS)
        _check_str(self, "reason")
        _check_str(self, "backbone", non_empty=True)
        _check_int(self, "seed")
        _check_str(self, "prompt_id", non_empty=True)


# --- JSONL I/O --------------------------------------------------------------


def _ends_without_newline(path: str) -> bool:
    """True if ``path`` exists, is non-empty, and its last byte is not ``\\n``."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return False
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"
    except FileNotFoundError:
        return False


def write_jsonl(path: str, records: Iterable[_Record], *, append: bool = False) -> int:
    """Write records to ``path`` as UTF-8 JSONL. Returns the number written.

    Overwrites by default; ``append=True`` adds to an existing file (for
    resumable experiment logs), first terminating an unterminated last line so
    new records never get glued onto it.
    """
    n = 0
    needs_newline = append and _ends_without_newline(path)
    with open(path, "a" if append else "w", encoding="utf-8", newline="\n") as f:
        if needs_newline:
            f.write("\n")
        for rec in records:
            f.write(rec.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str, cls: type[T]) -> Iterator[T]:
    """Yield records of type ``cls`` from a UTF-8 JSONL file (blank lines skipped).

    A leading UTF-8 BOM (e.g. from Excel or Windows PowerShell) is ignored.
    """
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield cls.from_json(line)
            except (SchemaError, json.JSONDecodeError) as e:
                raise SchemaError(f"{path}:{lineno}: {e}") from e
