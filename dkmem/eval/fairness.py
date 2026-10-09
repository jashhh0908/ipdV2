"""Comparability of run groups: do the systems being compared share inputs, model and settings?

``dkmem/tier1_eval_contract.md`` Sec 8 requires strategies that are compared to use the same backbone and model
settings (decoding, quantization, the embedding model behind ``similarity_score``) and any unavoidable difference
to be documented next to the comparison. ``check_comparable`` reads the ``run_config.json`` of each group (A-D,
store, Mem0 or flat RAG) and reports, per comparison key, every group pair that disagrees:

=====================  ===============================================================================
key                    where it comes from
=====================  ===============================================================================
``inputs``             ``input.inputs_sha256`` (hash of the ordered per-pair input hashes) and the
                       input file's sha256 when recorded
``seed``               ``seed``
``backbone``           ``backbone`` and ``model_run_info.model_config`` (model id, revision, dtype) and
                       the resolved weights revision; groups with no LLM (``backbone: null``, flat RAG,
                       Config D without ``lexicon+llm``) are exempt from the model keys, as the contract allows
``decoding``           ``generation_params`` without ``batch_size``
``embedder``           ``embedder.model_config`` (model id, revision, dtype) and ``similarity_metric``
                       (the one comparator behind ``similarity_score``)
``lexicon``            ``lexicon.sha256`` (groups that use a lexicon)
``pair_order``         the trace's ``eval_pair_id`` sequence
=====================  ===============================================================================

``batch_size`` is reported as a *note*, not an issue: it changes only how prompts are padded together (the Mem0
reproduction is sequential, Configs A-D batch 8) and is recorded so the report can state it. A key present in
only one group is not a mismatch (the other group does not use it).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from dkmem.eval.groups import GroupData

__all__ = ["comparability_key", "check_comparable"]


def _get(d: Mapping[str, Any] | None, *path: str) -> Any:
    for p in path:
        if not isinstance(d, Mapping):
            return None
        d = d.get(p)
    return d


def comparability_key(group: GroupData) -> dict[str, Any]:
    """The fields two groups must share to be compared (``None`` where a group does not use the field)."""
    cfg = group.run_config
    llm = cfg.get("backbone") is not None
    mc = _get(cfg, "model_run_info", "model_config") or {}
    gp = dict(cfg.get("generation_params") or {})
    batch_size = gp.pop("batch_size", None)
    emb_cfg = _get(cfg, "embedder", "model_config") or {}
    return {
        "inputs": {"inputs_sha256": _get(cfg, "input", "inputs_sha256"), "file_sha256": _get(cfg, "input", "file_sha256")},
        "seed": cfg.get("seed"),
        "backbone": {
            "backbone": cfg.get("backbone"),
            "model": {k: mc.get(k) for k in ("model_id", "revision", "dtype")} if llm else None,
            "resolved_revision": _get(cfg, "model_run_info", "resolved_revision") if llm else None,
        } if llm else None,
        "decoding": gp or None,
        "embedder": {
            "model": {k: emb_cfg.get(k) for k in ("model_id", "revision", "dtype")},
            "resolved_revision": _get(cfg, "embedder", "resolved_revision"),
            "similarity_metric": cfg.get("similarity_metric"),
        } if cfg.get("embedder") else None,
        "lexicon": _get(cfg, "lexicon", "sha256"),
        "pair_order": [t["eval_pair_id"] for t in group.trace],
        "_batch_size": batch_size,
    }


def check_comparable(groups: Sequence[GroupData]) -> dict[str, Any]:
    """``{"issues": [...], "notes": [...], "comparable": bool}`` for the groups (see the module docstring).

    Each issue names the key and the differing values with the groups that hold them."""
    keys = {g.group_id: comparability_key(g) for g in groups}
    issues: list[dict[str, Any]] = []
    notes: list[str] = []
    for key in ("inputs", "seed", "backbone", "decoding", "embedder", "lexicon", "pair_order"):
        by_value: dict[str, list[str]] = {}
        values: dict[str, Any] = {}
        for gid, k in keys.items():
            v = k[key]
            if v is None:
                continue
            if key == "inputs":
                v = {f: x for f, x in v.items() if x is not None}
            rep = repr(v)
            by_value.setdefault(rep, []).append(gid)
            values[rep] = v
        if len(by_value) > 1:
            issues.append({"key": key, "values": [{"value": values[r], "groups": gs} for r, gs in by_value.items()]})
    batch = {gid: k["_batch_size"] for gid, k in keys.items() if k["_batch_size"] is not None}
    if len(set(batch.values())) > 1:
        notes.append(f"batch_size differs (not an issue for greedy decoding, but report it): {batch}")
    return {"comparable": not issues, "issues": issues, "notes": notes}
