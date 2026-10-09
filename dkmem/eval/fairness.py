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
``pair_order``         the set of ``eval_pair_id`` the trace covers (kept under this name). Not the row
                       sequence: the A-D harness writes failed and skipped episodes before the decided ones,
                       so the sequence differs between configurations that fail on different episodes. The
                       input order itself is fixed by ``inputs.inputs_sha256``.
=====================  ===============================================================================

``batch_size`` is reported as a *note*, not an issue: it changes only how prompts are padded together (the Mem0
reproduction is sequential, Configs A-D batch 8) and is recorded so the report can state it. A key present in
only one group is not a mismatch (the other group does not use it).

Reproducibility checks (added with the multilingual evaluation audit). Beyond the keys above, ``check_comparable`` also
reports as *issues*

* ``lexicon_llm_policy``: groups that run mode ``lexicon+llm`` under different meanings of that mode (a run made before
  the 2026-10-08 policy change records none and counts as ``legacy``; its distinctions are not comparable to a new one);
* ``prompt_hash``: one prompt id recorded with two different sha256 (a "frozen" prompt that was edited);
* ``manifests``: a ``run_manifest.json`` or ``pairwise_eval.jsonl`` that contradicts its ``run_config.json``
  (``manifest_problems``);

and as *warnings* (they do not change ``comparable``, but a final run should have none): a model or embedder whose
weights revision was not recorded (``resolved_revision`` missing) or not pinned on the command line, a run made from a
dirty checkout, and no git commit recorded (as on Kaggle, where the per-file ``code`` hashes identify the code). Groups
built from different code trees and runs on a non-sanctioned input are *notes*.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from dkmem.eval.groups import GroupData, mode_slug

__all__ = ["comparability_key", "check_comparable", "manifest_problems", "pin_warnings"]


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
        "pair_order": sorted(t["eval_pair_id"] for t in group.trace),
        "_batch_size": batch_size,
    }


def manifest_problems(group: GroupData) -> list[str]:
    """Contradictions between a group's ``run_config.json`` and its per-mode ``run_manifest.json`` / rows."""
    cfg = group.run_config
    out: list[str] = []
    for mode in group.modes:
        try:
            m = group.manifest(mode)
            rows = group.rows(mode)
        except FileNotFoundError as e:
            out.append(f"{mode}: missing output file ({e.filename})")
            continue
        want = {"run_id": f"{group.group_id}-{mode_slug(mode)}", "seed": cfg.get("seed"), "backbone": cfg.get("backbone"),
                "dataset_tier": "tier1_minimal_pairs"}
        if "dkmem_mode" in m:
            want["dkmem_mode"] = mode
        if group.config_id is not None and "pipeline_config" in m:
            want["pipeline_config"] = group.config_id
        for k, v in want.items():
            if m.get(k) != v:
                out.append(f"{mode}: manifest {k} is {m.get(k)!r}, expected {v!r}")
        bad = {(r["run_id"], r["strategy"]) for r in rows} - {(m.get("run_id"), m.get("strategy"))}
        if bad:
            out.append(f"{mode}: {len(bad)} distinct (run_id, strategy) in pairwise_eval.jsonl differ from the manifest, e.g. {sorted(bad)[0]}")
    return out


def pin_warnings(group: GroupData) -> list[str]:
    """Weights, code and input identity that the group did not record or pin (see the module docstring)."""
    cfg = group.run_config
    out: list[str] = []
    for label, requested, resolved, used in (
        ("model", _get(cfg, "model_run_info", "model_config", "revision"), _get(cfg, "model_run_info", "resolved_revision"),
         cfg.get("backbone") is not None),
        ("embedder", _get(cfg, "embedder", "model_config", "revision"), _get(cfg, "embedder", "resolved_revision"),
         bool(cfg.get("embedder"))),
    ):
        if not used:
            continue
        if resolved is None:
            out.append(f"{group.group_id}: {label} weights revision not recorded (resolved_revision missing)")
        elif requested is None:
            out.append(f"{group.group_id}: {label} revision not pinned on the command line (resolved {resolved})")
        elif requested != resolved:
            out.append(f"{group.group_id}: {label} requested revision {requested} but resolved {resolved}")
    git = cfg.get("git") or {}
    if git.get("dirty"):
        out.append(f"{group.group_id}: run from a dirty git checkout")
    if git.get("commit") is None:
        out.append(f"{group.group_id}: no git commit recorded (identify the code by code.tree_sha256)")
    return out


def check_comparable(groups: Sequence[GroupData]) -> dict[str, Any]:
    """``{"issues": [...], "notes": [...], "warnings": [...], "comparable": bool}`` for the groups (see the module
    docstring).

    Each issue names the key and the differing values with the groups that hold them."""
    labels = _labels(groups)
    keys = {lab: comparability_key(g) for lab, g in zip(labels, groups)}
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

    policy = {lab: g.run_config.get("lexicon_llm_policy", "legacy") for lab, g in zip(labels, groups) if "lexicon+llm" in g.modes}
    if len(set(policy.values())) > 1:
        issues.append({"key": "lexicon_llm_policy", "values": _by_value(policy)})
    shas: dict[str, dict[str, list[str]]] = {}
    for lab, g in zip(labels, groups):
        for p in (g.run_config.get("prompts") or {}).values():
            shas.setdefault(p["prompt_id"], {}).setdefault(p["sha256"], []).append(lab)
    for pid, by_sha in sorted(shas.items()):
        if len(by_sha) > 1:
            issues.append({"key": "prompt_hash", "prompt_id": pid, "values": [{"sha256": s, "groups": gs} for s, gs in by_sha.items()]})
    for lab, g in zip(labels, groups):
        problems = manifest_problems(g)
        if problems:
            issues.append({"key": "manifests", "group": lab, "problems": problems})

    trees = {lab: (g.run_config.get("code") or {}).get("tree_sha256") for lab, g in zip(labels, groups)}
    if len(set(trees.values())) > 1:
        notes.append(f"groups were run from {len(set(trees.values()))} different code trees (code.tree_sha256): {trees}")
    unsanctioned = [lab for lab, g in zip(labels, groups) if (g.run_config.get("input") or {}).get("sanctioned") is False]
    if unsanctioned:
        notes.append(f"not run on the sanctioned Tier 1 input: {unsanctioned}")
    warnings = [w for g in groups for w in pin_warnings(g)]
    return {"comparable": not issues, "issues": issues, "notes": notes, "warnings": warnings}


def _labels(groups: Sequence[GroupData]) -> list[str]:
    """One unique label per group: its ``group_id``, with ``#<index>`` added when two groups share an id (e.g. a
    rerun and its original have the same fingerprint, hence the same id; keying by id would merge them and hide
    any difference between them)."""
    ids = [g.group_id for g in groups]
    return [gid if ids.count(gid) == 1 else f"{gid}#{i}" for i, gid in enumerate(ids)]


def _by_value(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for gid, v in values.items():
        out.setdefault(repr(v), {"value": v, "groups": []})["groups"].append(gid)
    return list(out.values())
