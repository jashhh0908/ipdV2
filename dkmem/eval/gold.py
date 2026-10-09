"""The gold relation of each Tier 1 pair (same entity / different entity).

Team A never sees the gold labels: the ``opaque_entity_id_*`` fields of the sanctioned input are all distinct
(they are passthrough ids, not labels), and the relation lives in Team B's answer key. Scoring therefore takes the
key as an explicit file chosen by whoever runs it; nothing in this repository reads a key by default, and the
runners never import this module.

File format (JSONL, one object per gold pair)::

    {"eval_pair_id": "ep_...", "relation": "same" | "different",
     "distinction_class": "kinship" | ... (optional), "language": "hi" | ... (optional)}

``gold_relation`` is accepted as an alias of ``relation``. Extra fields are ignored. Tier 1 v2 has 98 ``different``
and 75 ``same`` pairs (``dkmem/tier1_eval_contract.md`` Sec 5-6); the 53 excluded ambiguous candidates are simply
absent from the key and are not scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

__all__ = ["RELATIONS", "GoldError", "GoldPair", "load_gold", "gold_from_pairs"]

RELATIONS = ("same", "different")


class GoldError(ValueError):
    """The gold key is malformed or does not cover the pairs being scored."""


@dataclass(frozen=True)
class GoldPair:
    eval_pair_id: str
    relation: str
    distinction_class: str | None = None
    language: str | None = None

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
            pairs.append(GoldPair(obj["eval_pair_id"], relation, obj.get("distinction_class"), obj.get("language")))
        except (json.JSONDecodeError, KeyError, TypeError, GoldError) as e:
            raise GoldError(f"{path}:{i}: {e!r}") from e
    return gold_from_pairs(pairs)
