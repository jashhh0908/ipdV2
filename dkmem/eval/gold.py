"""The gold relation of each Tier 1 pair (same entity / different entity).

Team A never sees the gold labels: the ``opaque_entity_id_*`` fields of the sanctioned input are all distinct
(they are passthrough ids, not labels), and the relation lives in Team B's answer key. Scoring therefore takes the
key as an explicit file chosen by whoever runs it; nothing in this repository reads a key by default, and the
runners never import this module.

File format (JSONL, one object per gold pair)::

    {"eval_pair_id": "ep_...", "relation": "same" | "different",
     "distinction_class": "kinship" | ... (optional), "language": "hi" | ... (optional),
     "utterance_a_id", "utterance_b_id", "opaque_entity_id_a", "opaque_entity_id_b": (optional)}

``gold_relation`` is accepted as an alias of ``relation``. Other fields are ignored, except the four optional id
fields: when the key carries them (Team B's does), ``check_alignment`` verifies them against the input. Tier 1 v2 has 98 ``different``
and 75 ``same`` pairs (``dkmem/tier1_eval_contract.md`` Sec 5-6); the 53 excluded ambiguous candidates are simply
absent from the key and are not scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

__all__ = ["RELATIONS", "GoldError", "GoldPair", "load_gold", "gold_from_pairs", "check_alignment"]

RELATIONS = ("same", "different")


class GoldError(ValueError):
    """The gold key is malformed or does not cover the pairs being scored."""


@dataclass(frozen=True)
class GoldPair:
    eval_pair_id: str
    relation: str
    distinction_class: str | None = None
    language: str | None = None
    utterance_a_id: str | None = None
    utterance_b_id: str | None = None
    opaque_entity_id_a: str | None = None
    opaque_entity_id_b: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.eval_pair_id, str) or not self.eval_pair_id:
            raise GoldError(f"eval_pair_id must be a non-empty string, got {self.eval_pair_id!r}")
        if self.relation not in RELATIONS:
            raise GoldError(f"{self.eval_pair_id}: relation must be one of {RELATIONS}, got {self.relation!r}")


def gold_from_pairs(pairs: Iterable[GoldPair]) -> dict[str, GoldPair]:
    out: dict[str, GoldPair] = {}
    for p in pairs:
        if p.eval_pair_id in out:
            raise GoldError(f"duplicate gold pair {p.eval_pair_id!r}")
        out[p.eval_pair_id] = p
    return out


def load_gold(path: str | Path) -> dict[str, GoldPair]:
    """Parse a gold key file (see the module docstring)."""
    pairs = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj: Mapping = json.loads(line)
            relation = obj["relation"] if "relation" in obj else obj["gold_relation"]
            pairs.append(
                GoldPair(
                    obj["eval_pair_id"], relation, obj.get("distinction_class"), obj.get("language"),
                    obj.get("utterance_a_id"), obj.get("utterance_b_id"),
                    obj.get("opaque_entity_id_a"), obj.get("opaque_entity_id_b"),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, GoldError) as e:
            raise GoldError(f"{path}:{i}: {e!r}") from e
    return gold_from_pairs(pairs)


_ALIGNED_FIELDS = ("language", "utterance_a_id", "utterance_b_id", "opaque_entity_id_a", "opaque_entity_id_b")


def check_alignment(gold: Mapping[str, GoldPair], records: Iterable) -> None:
    """Raise ``GoldError`` if the key disagrees with the input file it is meant to score.

    For every key pair that carries the optional fields (``language``, the two utterance ids and the two opaque entity
    ids), they must equal the input record's with the same ``eval_pair_id``; a key pair with no record is an error.
    A key made for another regeneration of the input (the contract's ``--regenerate``) would otherwise be joined by
    ``eval_pair_id`` alone. Pairs of the input that the key does not list (the contract's excluded ones) are fine.
    """
    by_id = {r.eval_pair_id: r for r in records}
    problems = []
    for pid, g in gold.items():
        r = by_id.get(pid)
        if r is None:
            problems.append(f"{pid}: in the key but not in the input")
            continue
        for f in _ALIGNED_FIELDS:
            want = getattr(g, f)
            if want is not None and want != getattr(r, f):
                problems.append(f"{pid}: {f} is {want!r} in the key but {getattr(r, f)!r} in the input")
    if problems:
        raise GoldError(f"{len(problems)} key/input disagreement(s), e.g. " + "; ".join(problems[:3]))
