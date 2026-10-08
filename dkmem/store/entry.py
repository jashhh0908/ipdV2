"""Data records of the DK-Mem memory store.

``StoredEntry`` is one memory the store holds. It keeps what DK-Mem retains per
entry -- ``stored_text`` (the config's gloss / native fact / utterance), the
verbatim ``surface`` and the write-side ``distinction`` -- plus the provenance a
merge leaves behind (``members``, ``history``). It carries no gold label: gold
entity ids are joined from the input file at analysis time and never read by the
store, the host or the gate.

``Link`` is an unresolved link between two active entries. Links are
logging-only for now: nothing in retrieval or consolidation reads them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

__all__ = ["ENTRY_STATUSES", "LINK_KINDS", "StoredEntry", "Link", "member_record"]

ENTRY_STATUSES = ("active", "absorbed")
LINK_KINDS = ("unresolved",)


@dataclass
class StoredEntry:
    """One memory entry.

    ``distinction`` is the dict the gate sees for this entry in the run's DK-Mem
    mode (``{}`` = nothing marked; ``None`` = unavailable, e.g. no V4 extraction
    in mode ``lexicon+llm``; such an entry is never written in a gated mode, see
    ``dkmem.store.consolidate``).

    ``members`` lists every entry merged into this one, itself first, each with the
    ``distinction`` it brought. ``history`` lists the texts a merge replaced
    (newer text wins; the old one stays here, readable by audit but not by
    retrieval or the judge). An ``absorbed`` entry is a tombstone: it was merged
    into ``absorbed_by`` at its own write and is never retrievable.
    """

    entry_id: str
    conv_id: str
    source_utterance_id: str
    language: str
    stored_text: str
    surface: str
    distinction: dict[str, str] | None
    session_id: str | None = None
    turn_idx: int | None = None
    seq: int | None = None
    members: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    status: str = "active"
    absorbed_by: str | None = None

    def __post_init__(self) -> None:
        for name in ("entry_id", "conv_id", "source_utterance_id", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"StoredEntry.{name} must be a non-empty string, got {value!r}")
        for name in ("stored_text", "surface"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"StoredEntry.{name} must be a string")
        if self.distinction is not None:
            if not isinstance(self.distinction, Mapping):
                raise ValueError("StoredEntry.distinction must be a mapping or None")
            self.distinction = dict(self.distinction)
        if self.status not in ENTRY_STATUSES:
            raise ValueError(f"StoredEntry.status must be one of {ENTRY_STATUSES}, got {self.status!r}")

    @property
    def member_utterance_ids(self) -> set[str]:
        """Source utterances whose content is in this entry (its own, plus merged-in ones)."""
        return {self.source_utterance_id} | {m["source_utterance_id"] for m in self.members}

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable copy."""
        return asdict(self)


def member_record(entry: StoredEntry) -> dict[str, Any]:
    """The ``members`` item describing ``entry`` as a contributor to a cluster."""
    return {
        "entry_id": entry.entry_id,
        "source_utterance_id": entry.source_utterance_id,
        "surface": entry.surface,
        "distinction": None if entry.distinction is None else dict(entry.distinction),
        "seq": entry.seq,
    }


@dataclass(frozen=True)
class Link:
    """An undirected, non-binding "may be the same" link between two active entries.

    Created only when the host proposed a merge and the gate changed it to
    ``link_unresolved`` (the pair is ``underdetermined``). ``a``/``b`` are stored in sorted order.
    """

    a: str
    b: str
    seq: int
    reason: str
    kind: str = "unresolved"

    def __post_init__(self) -> None:
        if self.kind not in LINK_KINDS:
            raise ValueError(f"Link.kind must be one of {LINK_KINDS}, got {self.kind!r}")
        if self.a == self.b:
            raise ValueError("a link needs two different entries")
        if self.a > self.b:
            a, b = self.b, self.a
            object.__setattr__(self, "a", a)
            object.__setattr__(self, "b", b)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
