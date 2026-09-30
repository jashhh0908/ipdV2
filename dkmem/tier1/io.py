"""Team B Tier 1 input/output contract: loading the sanctioned input file and
writing ``run_manifest.json`` / ``pairwise_eval.jsonl`` exactly as
``teamA_tier1_run_instructions.md`` / ``teamA_tier1_handoff(1).md`` require.

Input
-----
``load_tier1_input`` reads *only* the one file Team A is allowed to use
(``data/tier1/eval_input/teamA_tier1_v2.jsonl``), verifies its sha256 and
record count before returning anything, and parses each line into a
``Tier1Record`` holding exactly the 8 sanctioned fields. Nothing in this
module (or anywhere in ``dkmem.tier1``) opens any of the files Team B
forbids: ``data/tier1/candidates.jsonl``, ``data/tier1/annotations/``,
``data/tier1/tier1_eval.jsonl``, ``data/tier1/tier1_eval_v2.jsonl``,
``data/tier1/eval_input/teamB_answer_key.jsonl``.

``Tier1Record.opaque_entity_id_a/b`` are carried only to be copied verbatim
into a produced entry's ``gold_entity_id`` on output -- they are never read
by extraction, the lexicon, the gate, or the similarity step, and nothing in
this package treats them as a feature.

Output
------
``PairwiseEvalRecord``/``MemoryEntry`` are Team B's external contract
(``pairwise_eval.schema.json``), a schema deliberately separate from
``dkmem.memory.schema.MergeEvent``/``Extraction`` (both unchanged): the two
use different decision vocabularies, different distinction shapes, and
``MergeEvent`` has no ``compatibility`` field, so translating between them,
rather than reusing one for the other, is the point of this module.

Decision translation (pairwise_eval.schema.json ``decision`` enum): DK-Mem's
internal ``keep_both``/``link_unresolved`` (``dkmem.memory.schema.
DECISIONS``) become Team B's ``no_merge``/``underdetermined_link``; ``merge``
passes through unchanged. ``supersede`` is accepted on the output side but
nothing in ``dkmem.tier1`` ever produces it, since ``dkmem.memory.gate``
does not implement supersede detection.

Distinction translation (``MemoryEntry.distinction``): the external schema
wants *one* ``{class, value, script, extraction_method}`` object per entry,
using its own 7-class enum, not ``Extraction.distinction``'s internal
``dict[str, str]`` (which can hold several class/value pairs at once, and
uses different names for 4 of the 7 classes: ``register`` ->
``honorific_register``, ``classifier`` -> ``classifier_measure``,
``politeness`` -> ``politeness_relationship``, ``temporal_deixis`` ->
``spatial_temporal_deixis``; ``kinship``/``evidentiality``/``name_variant``
are unchanged). ``dkmem.memory.gate`` keeps gating on the full internal
dict, unchanged -- this translation only affects what gets *reported* per
entry. When an entry's distinction dict has more than one key (e.g. a
sentence marking both kinship and register), only one is reported, chosen
deterministically (the alphabetically-first internal key) -- a limitation
of this serialization, not of the gate's own (unaffected) decision, which
still considers every key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dkmem.memory.gate import COMPATIBILITY

__all__ = [
    "TIER1_INPUT_SHA256",
    "TIER1_INPUT_RECORD_COUNT",
    "TIER1_STRATEGIES",
    "PAIRWISE_DECISIONS",
    "DECISION_TRANSLATION",
    "Tier1InputError",
    "Tier1Record",
    "load_tier1_input",
    "DISTINCTION_CLASSES",
    "DISTINCTION_CLASS_TRANSLATION",
    "EXTRACTION_METHODS",
    "DistinctionMarker",
    "translate_distinction",
    "MemoryEntry",
    "PairwiseEvalRecord",
    "write_run_manifest",
    "write_pairwise_eval",
]

TIER1_INPUT_SHA256 = "eab5dcb23375eb2e744c72a98cb31c25da0171b2ba3297a0843574ed9a34cd07"
TIER1_INPUT_RECORD_COUNT = 173

# This task's scope only; the full Team B list has 9 strategies.
TIER1_STRATEGIES = ("dk-mem-lexicon", "dk-mem-lexicon-llm")

PAIRWISE_DECISIONS = ("merge", "no_merge", "supersede", "underdetermined_link")

# dkmem.memory.schema.DECISIONS -> Team B's pairwise_eval.jsonl vocabulary.
DECISION_TRANSLATION = {
    "merge": "merge",
    "supersede": "supersede",
    "keep_both": "no_merge",
    "link_unresolved": "underdetermined_link",
}

# pairwise_eval.schema.json's MemoryEntry.distinction.class enum.
DISTINCTION_CLASSES = (
    "kinship",
    "honorific_register",
    "name_variant",
    "evidentiality",
    "classifier_measure",
    "politeness_relationship",
    "spatial_temporal_deixis",
)

# Internal distinction key (dkmem.memory.extract.DISTINCTION_KEYS /
# dkmem.memory.gate.DEFAULT_DISCRIMINATIVE_FEATURES) -> external class name.
# A 1:1, onto mapping: every internal key has exactly one external name.
DISTINCTION_CLASS_TRANSLATION = {
    "kinship": "kinship",
    "register": "honorific_register",
    "name_variant": "name_variant",
    "evidentiality": "evidentiality",
    "classifier": "classifier_measure",
    "politeness": "politeness_relationship",
    "temporal_deixis": "spatial_temporal_deixis",
}

EXTRACTION_METHODS = ("lexicon", "lexicon+llm")

_TIER1_RECORD_FIELDS = frozenset(
    {
        "eval_pair_id",
        "language",
        "utterance_a",
        "utterance_b",
        "utterance_a_id",
        "utterance_b_id",
        "opaque_entity_id_a",
        "opaque_entity_id_b",
    }
)


class Tier1InputError(ValueError):
    """The Tier 1 input file, or one of its records, fails validation."""


def _nonempty_str(obj: Any, name: str, error: type[Exception]) -> None:
    value = getattr(obj, name)
    if not isinstance(value, str) or not value.strip():
        raise error(f"{type(obj).__name__}.{name} must be a non-empty string, got {value!r}")


@dataclass(frozen=True)
class Tier1Record:
    """One of the 173 sanctioned Team B input records, unmodified.

    ``opaque_entity_id_a``/``opaque_entity_id_b`` are passthrough-only: carry
    them to the output's ``gold_entity_id`` field, never read them as input
    to extraction, matching, gating, or similarity.
    """

    eval_pair_id: str
    language: str
    utterance_a: str
    utterance_b: str
    utterance_a_id: str
    utterance_b_id: str
    opaque_entity_id_a: str
    opaque_entity_id_b: str

    def __post_init__(self) -> None:
        for name in (
            "eval_pair_id", "language", "utterance_a", "utterance_b",
            "utterance_a_id", "utterance_b_id", "opaque_entity_id_a", "opaque_entity_id_b",
        ):
            _nonempty_str(self, name, Tier1InputError)
        for side, uid in (("a", self.utterance_a_id), ("b", self.utterance_b_id)):
            if not uid.startswith(self.eval_pair_id):
                raise Tier1InputError(
                    f"utterance_{side}_id {uid!r} does not start with eval_pair_id "
                    f"{self.eval_pair_id!r}"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tier1Record":
        if not isinstance(data, dict):
            raise Tier1InputError(f"record must be an object, got {type(data).__name__}")
        keys = set(data)
        if keys != _TIER1_RECORD_FIELDS:
            missing = _TIER1_RECORD_FIELDS - keys
            unknown = keys - _TIER1_RECORD_FIELDS
            raise Tier1InputError(f"record fields: missing {sorted(missing)}, unexpected {sorted(unknown)}")
        try:
            return cls(**data)
        except TypeError as e:
            raise Tier1InputError(str(e)) from e


def load_tier1_input(path: str | Path) -> list[Tier1Record]:
    """Load, hash-check, and parse the sanctioned Tier 1 input file.

    Raises ``Tier1InputError`` if the file's sha256 doesn't match
    ``TIER1_INPUT_SHA256``, if it doesn't have exactly
    ``TIER1_INPUT_RECORD_COUNT`` records, or if any record is malformed. Per
    the run instructions: "Check the hash before every run. If it doesn't
    match, stop and ask Team B" -- this function enforces that rather than
    leaving it to the caller.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise Tier1InputError(f"{path}: {e}") from e

    digest = hashlib.sha256(raw).hexdigest()
    if digest != TIER1_INPUT_SHA256:
        raise Tier1InputError(
            f"{path}: sha256 {digest} does not match expected {TIER1_INPUT_SHA256}; "
            "stop and ask Team B, per the run instructions"
        )

    lines = [l for l in raw.decode("utf-8").splitlines() if l.strip()]
    if len(lines) != TIER1_INPUT_RECORD_COUNT:
        raise Tier1InputError(
            f"{path}: expected {TIER1_INPUT_RECORD_COUNT} records, found {len(lines)}"
        )

    records = []
    for i, line in enumerate(lines):
        try:
            records.append(Tier1Record.from_dict(json.loads(line)))
        except (json.JSONDecodeError, Tier1InputError) as e:
            raise Tier1InputError(f"{path}:{i + 1}: {e}") from e
    return records


@dataclass(frozen=True)
class DistinctionMarker:
    """``MemoryEntry.distinction``: one discriminative marker, external shape.

    ``script`` is left ``None`` (schema-valid: optional/nullable) rather than
    guessed -- ``dkmem.memory.matcher.match_distinctions`` does not currently
    report which script matched, and this module does not infer one.
    """

    cls: str
    value: str
    script: str | None = None
    extraction_method: str | None = None

    def __post_init__(self) -> None:
        if self.cls not in DISTINCTION_CLASSES:
            raise Tier1InputError(f"distinction class must be one of {DISTINCTION_CLASSES}, got {self.cls!r}")
        _nonempty_str(self, "value", Tier1InputError)
        if self.script is not None and (not isinstance(self.script, str) or not self.script.strip()):
            raise Tier1InputError(f"script must be None or a non-empty string, got {self.script!r}")
        if self.extraction_method is not None and self.extraction_method not in EXTRACTION_METHODS:
            raise Tier1InputError(
                f"extraction_method must be None or one of {EXTRACTION_METHODS}, "
                f"got {self.extraction_method!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        # The schema's field is named "class", a Python keyword; the
        # dataclass field is "cls" and renamed only at serialization.
        return {
            "class": self.cls,
            "value": self.value,
            "script": self.script,
            "extraction_method": self.extraction_method,
        }


def translate_distinction(
    distinction: dict[str, str], *, extraction_method: str | None
) -> DistinctionMarker | None:
    """One internal ``Extraction.distinction`` dict -> one external marker.

    ``None`` if ``distinction`` is empty. Otherwise picks the
    alphabetically-first internal key (deterministic; see the module
    docstring for why only one is reported when several are present) and
    translates its name via ``DISTINCTION_CLASS_TRANSLATION``.
    """
    if not distinction:
        return None
    key = min(distinction)
    if key not in DISTINCTION_CLASS_TRANSLATION:
        raise Tier1InputError(f"no external class mapping for internal distinction key {key!r}")
    return DistinctionMarker(
        cls=DISTINCTION_CLASS_TRANSLATION[key],
        value=distinction[key],
        extraction_method=extraction_method,
    )


@dataclass(frozen=True)
class MemoryEntry:
    """One ``entry_a``/``entry_b`` object inside a ``PairwiseEvalRecord``."""

    entry_id: str
    source_utterance_id: str
    gold_entity_id: str
    language: str
    gloss: str
    surface: str | None = None
    distinction: DistinctionMarker | None = None

    def __post_init__(self) -> None:
        for name in ("entry_id", "source_utterance_id", "gold_entity_id", "language", "gloss"):
            _nonempty_str(self, name, Tier1InputError)
        if self.surface is not None and not isinstance(self.surface, str):
            raise Tier1InputError(f"surface must be None or a string, got {type(self.surface).__name__}")
        if self.distinction is not None and not isinstance(self.distinction, DistinctionMarker):
            raise Tier1InputError("distinction must be None or a DistinctionMarker")

    def to_dict(self) -> dict[str, Any]:
        d = {
            "entry_id": self.entry_id,
            "source_utterance_id": self.source_utterance_id,
            "gold_entity_id": self.gold_entity_id,
            "language": self.language,
            "gloss": self.gloss,
            "surface": self.surface,
            "distinction": self.distinction.to_dict() if self.distinction is not None else None,
        }
        return d


@dataclass(frozen=True)
class PairwiseEvalRecord:
    """One line of Team B's ``pairwise_eval.jsonl``.

    ``similarity_score`` must be the raw, pre-gating gloss similarity
    (never clipped, rounded, or replaced by a post-gating value);
    ``threshold`` is the tau actually used; ``compatibility`` is one of
    ``dkmem.memory.gate.COMPATIBILITY`` for a gated strategy, ``None`` for
    one without gating.
    """

    record_id: str
    run_id: str
    strategy: str
    entry_a: MemoryEntry
    entry_b: MemoryEntry
    decision: str
    similarity_score: float
    threshold: float
    compatibility: str | None
    predicted_entity_id: str | None = None
    superseded_entry_id: str | None = None

    def __post_init__(self) -> None:
        _nonempty_str(self, "record_id", Tier1InputError)
        _nonempty_str(self, "run_id", Tier1InputError)
        if self.strategy not in TIER1_STRATEGIES:
            raise Tier1InputError(f"strategy must be one of {TIER1_STRATEGIES}, got {self.strategy!r}")
        if not isinstance(self.entry_a, MemoryEntry) or not isinstance(self.entry_b, MemoryEntry):
            raise Tier1InputError("entry_a/entry_b must be MemoryEntry instances")
        if self.decision not in PAIRWISE_DECISIONS:
            raise Tier1InputError(f"decision must be one of {PAIRWISE_DECISIONS}, got {self.decision!r}")
        # pairwise_eval.schema.json: threshold/similarity_score in [0, 1].
        for name in ("similarity_score", "threshold"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise Tier1InputError(f"{name} must be a number")
            if not (0 <= value <= 1):
                raise Tier1InputError(f"{name} must be in [0, 1], got {value!r}")
        if self.compatibility is not None and self.compatibility not in COMPATIBILITY:
            raise Tier1InputError(f"compatibility must be None or one of {COMPATIBILITY}")

        wants_entity_id = self.decision in ("merge", "supersede")
        if wants_entity_id and self.predicted_entity_id is None:
            raise Tier1InputError(f"predicted_entity_id must be set when decision={self.decision!r}")
        if not wants_entity_id and self.predicted_entity_id is not None:
            raise Tier1InputError(f"predicted_entity_id must be null when decision={self.decision!r}")

        if self.decision != "supersede" and self.superseded_entry_id is not None:
            raise Tier1InputError("superseded_entry_id must be null unless decision='supersede'")
        if self.decision == "supersede" and self.superseded_entry_id not in (
            self.entry_a.entry_id,
            self.entry_b.entry_id,
        ):
            raise Tier1InputError("superseded_entry_id must equal entry_a.entry_id or entry_b.entry_id")

    def to_dict(self) -> dict[str, Any]:
        # Not dataclasses.asdict(): MemoryEntry/DistinctionMarker need their
        # own to_dict() (the "class" field can't be a Python field name).
        return {
            "record_id": self.record_id,
            "run_id": self.run_id,
            "strategy": self.strategy,
            "entry_a": self.entry_a.to_dict(),
            "entry_b": self.entry_b.to_dict(),
            "decision": self.decision,
            "similarity_score": self.similarity_score,
            "threshold": self.threshold,
            "compatibility": self.compatibility,
            "predicted_entity_id": self.predicted_entity_id,
            "superseded_entry_id": self.superseded_entry_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


def write_run_manifest(
    path: str | Path,
    *,
    run_id: str,
    strategy: str,
    seed: int,
    backbone: str | None,
    created_at: str | None = None,
) -> None:
    """Write ``run_manifest.json``. ``backbone`` is ``None`` (JSON ``null``)
    only for a strategy with no LLM stage.
    """
    if strategy not in TIER1_STRATEGIES:
        raise Tier1InputError(f"strategy must be one of {TIER1_STRATEGIES}, got {strategy!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise Tier1InputError("seed must be an int")
    if created_at is not None:
        try:
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as e:
            raise Tier1InputError(f"created_at must be an RFC 3339 date-time, got {created_at!r}: {e}") from e
    manifest = {
        "run_id": run_id,
        "strategy": strategy,
        "dataset_tier": "tier1_minimal_pairs",
        "seed": seed,
        "backbone": backbone,
    }
    if created_at is not None:
        manifest["created_at"] = created_at
    Path(path).write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )


def write_pairwise_eval(path: str | Path, records: list[PairwiseEvalRecord]) -> int:
    """Write ``pairwise_eval.jsonl``: one line per record actually made.

    Raises ``Tier1InputError`` if any ``record_id`` repeats (checked before
    anything is written). Returns the number of lines written.
    """
    seen = set()
    for r in records:
        if r.record_id in seen:
            raise Tier1InputError(f"duplicate record_id: {r.record_id!r}")
        seen.add(r.record_id)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(r.to_json() + "\n")
    return len(records)
