"""Loading a run group directory (A-D, store, Mem0 or flat-RAG output) for offline evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dkmem.eval.metrics import read_jsonl

__all__ = ["GroupData", "load_group", "mode_slug"]


def mode_slug(mode: str) -> str:
    return mode.replace("+", "-")


@dataclass
class GroupData:
    """One run group: ``run_config.json``, ``trace.jsonl`` and, per DK-Mem mode, ``pairwise_eval.jsonl``."""

    path: Path
    run_config: dict[str, Any]
    trace: list[dict[str, Any]]
    _rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def group_id(self) -> str:
        return self.run_config["group_id"]

    @property
    def modes(self) -> list[str]:
        return list(self.run_config["dkmem_modes"])

    @property
    def config_id(self) -> str | None:
        pc = self.run_config.get("pipeline_config")
        return pc["config_id"] if isinstance(pc, dict) else None

    @property
    def strategy_label(self) -> str:
        """Short human label: pipeline config id, or the baseline name for Mem0 / flat RAG."""
        return self.config_id or (self.run_config.get("baseline") or {}).get("name", "?")

    def rows(self, mode: str) -> list[dict[str, Any]]:
        if mode not in self._rows:
            self._rows[mode] = read_jsonl(self.path / mode_slug(mode) / "pairwise_eval.jsonl")
        return self._rows[mode]

    def manifest(self, mode: str) -> dict[str, Any]:
        return json.loads((self.path / mode_slug(mode) / "run_manifest.json").read_text(encoding="utf-8"))

    def no_entry_ids(self) -> set[str]:
        """Pairs an utterance of which produced zero entries (A/B extraction failure, Mem0 ``no_entry``)."""
        return {t["eval_pair_id"] for t in self.trace if t.get("status") in ("extraction_failed", "no_entry")}


def load_group(path: str | Path) -> GroupData:
    path = Path(path)
    run_config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    return GroupData(path, run_config, read_jsonl(path / "trace.jsonl"))
