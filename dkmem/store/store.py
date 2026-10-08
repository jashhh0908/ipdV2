"""The in-memory DK-Mem memory store (one conversation).

A plain Python object: ``entries`` (``entry_id`` -> ``StoredEntry``), the unresolved
``links``, and an append-only ``events`` list. No database. It knows nothing about
hosts, the gate, similarity or gold labels: it only applies the three write
operations the consolidation flow chooses between.

====================  ========================================================
operation             effect
====================  ========================================================
``add``               a new active entry (host ``keep_both``, or no candidate)
``absorb``            a merge: the new entry is tombstoned, the target takes the
                      newer text, the old text moves to ``history``, the new
                      entry's provenance joins ``members``
``link``              an unresolved link between two active entries
====================  ========================================================

Every operation appends an event (``{"event": "add" | "absorb" | "link", ...}``)
so the final state can be replayed from the log. ``dkmem.store.consolidate``
appends one further ``write`` / ``write_skipped`` event per write, describing the
whole decision.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from dkmem.memory.gate import DEFAULT_DISCRIMINATIVE_FEATURES, compatible
from dkmem.store.entry import Link, StoredEntry, member_record

__all__ = ["StoreError", "MemoryStore"]


class StoreError(ValueError):
    """An operation that would corrupt the store (duplicate id, absorbing into a tombstone, ...)."""


class MemoryStore:
    """Entries, links and the event log of one conversation."""

    def __init__(self, conv_id: str, *, meta: Mapping[str, Any] | None = None) -> None:
        if not isinstance(conv_id, str) or not conv_id.strip():
            raise StoreError(f"conv_id must be a non-empty string, got {conv_id!r}")
        self.conv_id = conv_id
        self.meta: dict[str, Any] = dict(meta or {})
        self.entries: dict[str, StoredEntry] = {}
        self.links: list[Link] = []
        self.events: list[dict[str, Any]] = []
        self._writes = 0

    # --- bookkeeping -----------------------------------------------------------

    def next_write_index(self) -> int:
        """Counter of writes attempted so far (also for skipped ones); used as ``seq``."""
        idx = self._writes
        self._writes += 1
        return idx

    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append ``event`` to the log (numbered); returns it."""
        event = {"n": len(self.events), **event}
        self.events.append(event)
        return event

    # --- reads -------------------------------------------------------------------

    def get(self, entry_id: str) -> StoredEntry:
        try:
            return self.entries[entry_id]
        except KeyError:
            raise StoreError(f"no entry {entry_id!r} in store {self.conv_id!r}") from None

    def active(self) -> list[StoredEntry]:
        """Active entries in insertion order."""
        return [e for e in self.entries.values() if e.status == "active"]

    def links_of(self, entry_id: str) -> list[Link]:
        return [l for l in self.links if entry_id in (l.a, l.b)]

    # --- writes --------------------------------------------------------------------

    def _check_new(self, entry: StoredEntry) -> None:
        if entry.entry_id in self.entries:
            raise StoreError(f"entry_id {entry.entry_id!r} already in store {self.conv_id!r}")
        if entry.conv_id != self.conv_id:
            raise StoreError(f"entry {entry.entry_id!r} belongs to conversation {entry.conv_id!r}, not {self.conv_id!r}")
        if entry.status != "active" or entry.absorbed_by is not None or entry.history:
            raise StoreError(f"entry {entry.entry_id!r} must be a fresh active entry")

    def add(self, entry: StoredEntry) -> StoredEntry:
        """Store ``entry`` as a new active entry."""
        self._check_new(entry)
        entry.members = [member_record(entry)]
        self.entries[entry.entry_id] = entry
        self.record({"event": "add", "entry_id": entry.entry_id, "seq": entry.seq})
        return entry

    def absorb(self, target_id: str, entry: StoredEntry, *, decision: str = "merge") -> StoredEntry:
        """Merge ``entry`` into the active entry ``target_id`` (newer text wins).

        ``entry`` is kept as an ``absorbed`` tombstone (so its id still resolves); the
        target's discriminative distinction is left unchanged.
        """
        if decision not in ("merge", "supersede"):
            raise StoreError(f"absorb decision must be merge or supersede, got {decision!r}")
        target = self.get(target_id)
        if target.status != "active":
            raise StoreError(f"cannot absorb into {target_id!r}: status is {target.status!r}")
        self._check_new(entry)
        own = member_record(entry)
        entry.members = [dict(own)]
        entry.status, entry.absorbed_by = "absorbed", target_id
        self.entries[entry.entry_id] = entry
        target.history.append(
            {
                "replaced_text": target.stored_text,
                "replaced_surface": target.surface,
                "by_entry_id": entry.entry_id,
                "seq": entry.seq,
                "decision": decision,
            }
        )
        target.stored_text, target.surface = entry.stored_text, entry.surface
        target.members.append(own)
        self.record(
            {"event": "absorb", "entry_id": entry.entry_id, "target_id": target_id, "decision": decision, "seq": entry.seq}
        )
        return target

    def link(self, a: str, b: str, *, seq: int, reason: str) -> bool:
        """Add an unresolved link between two active entries; ``False`` if that pair is already linked."""
        ea, eb = self.get(a), self.get(b)
        for e in (ea, eb):
            if e.status != "active":
                raise StoreError(f"cannot link {e.entry_id!r}: status is {e.status!r}")
        link = Link(a, b, seq, reason)
        new = not any(l.a == link.a and l.b == link.b for l in self.links)
        if new:
            self.links.append(link)
        self.record({"event": "link", "a": link.a, "b": link.b, "seq": seq, "reason": reason, "new": new})
        return new

    # --- reporting -----------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Counts for bloat reporting."""
        active = self.active()
        return {
            "entries_total": len(self.entries),
            "entries_active": len(active),
            "entries_absorbed": len(self.entries) - len(active),
            "merged_clusters": sum(1 for e in active if len(e.members) > 1),
            "unresolved_links": len(self.links),
            "writes": self._writes,
            "writes_skipped": sum(1 for ev in self.events if ev["event"] == "write_skipped"),
        }

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable state: entries (all statuses), links, stats, meta."""
        return {
            "conv_id": self.conv_id,
            "meta": dict(self.meta),
            "stats": self.stats(),
            "entries": [e.to_dict() for e in self.entries.values()],
            "links": [l.to_dict() for l in self.links],
        }

    # --- checks --------------------------------------------------------------------

    def integrity_errors(self) -> list[str]:
        """Structural problems (an empty list means sound): independent of the gate or mode."""
        errors: list[str] = []
        for eid, e in self.entries.items():
            if e.entry_id != eid:
                errors.append(f"{eid}: key does not match entry_id {e.entry_id}")
            if e.status == "absorbed":
                target = self.entries.get(e.absorbed_by or "")
                if target is None or target.status != "active":
                    errors.append(f"{eid}: absorbed into {e.absorbed_by!r}, which is not an active entry")
                elif eid not in {m["entry_id"] for m in target.members}:
                    errors.append(f"{eid}: absorbed into {target.entry_id} but not in its members")
            else:
                if e.absorbed_by is not None:
                    errors.append(f"{eid}: active but has absorbed_by")
                if not e.members or e.members[0]["entry_id"] != eid:
                    errors.append(f"{eid}: members must start with the entry itself")
        seen = set()
        for l in self.links:
            for end in (l.a, l.b):
                if end not in self.entries or self.entries[end].status != "active":
                    errors.append(f"link {l.a}~{l.b}: {end} is not an active entry")
            if (l.a, l.b) in seen:
                errors.append(f"link {l.a}~{l.b} duplicated")
            seen.add((l.a, l.b))
        return errors

    def gate_conflicts(self, discriminative_features: Iterable[str] = DEFAULT_DISCRIMINATIVE_FEATURES) -> list[str]:
        """Clusters whose members disagree on, or only partly mark, a discriminative feature.

        With the DK-Mem gate on, this must be empty: a merge needs ``compatible``, and a
        compatible merge never changes the discriminative marks. It is expected to be
        non-empty in mode ``off`` (that is the false consolidation being measured).
        Members with an unavailable (``None``) distinction are skipped.
        """
        features = tuple(discriminative_features)
        conflicts = []
        for e in self.active():
            members = [m for m in e.members if m["distinction"] is not None]
            for i, m in enumerate(members):
                for other in members[i + 1:]:
                    verdict = compatible(m["distinction"], other["distinction"], features)
                    if verdict != "compatible":
                        conflicts.append(f"{e.entry_id}: {m['entry_id']} vs {other['entry_id']} are {verdict}")
        return conflicts
