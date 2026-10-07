"""Hashing, provenance and loss-point diagnosis for the stage-attribution harness.

Everything here is deterministic and model-free. The trace record layout is
documented in ``dkmem.pipeline.runner``.

Loss-point diagnosis
--------------------
Gold identity labels are withheld from Team A, so the diagnosis cannot say
whether a pair *should* merge. It asks a narrower, checkable question using the
DK-Mem lexicon as the yardstick: does the lexicon see a discriminative
difference between the two utterances (``lexicon_compatibility`` is
``incompatible`` or ``underdetermined``), and if so, at which stage did that
difference stop being visible to the merge decision?

- The lexicon is run on the utterance (the source of the difference) and again
  on each *stored* text. A distinction is *visible in storage* when the lexicon
  finds the same value in the stored text. English glosses of Hindi kinship
  terms fail this test; verbatim text passes it by construction.
- ``first_loss_point``:
  ``no_detectable_difference`` (nothing for the lexicon to preserve);
  ``dropped_at_extraction`` (no entry was stored for a side, so there was nothing
  to merge or keep apart; see ``diagnose_dropped``);
  ``storage`` (the differing distinction is no longer visible in a stored text
  **and the two stored texts have collapsed into the same text**: L1 in Config A,
  L2 in B);
  ``host_merge`` (the distinction survived storage, or at least the stored texts
  still differ, yet the host proposed a merge: L3 judge, L4 threshold);
  ``none`` (the host kept the entries apart).
  ``storage_state`` says which storage case applied: ``visible`` (the lexicon
  still finds the differing values in the stored texts), ``collapsed`` (not
  visible and the texts are identical after normalization: actual loss) or
  ``distinct_not_visible`` (not visible, but the texts still differ, e.g. *mami*
  glossed "mother" and *mausi* glossed "aunt": the term was degraded, not erased,
  and the host could still tell the entries apart). Only ``collapsed`` is reported
  as a storage loss; a ``distinct_not_visible`` pair falls through to the host check.

The diagnosis covers only what the lexicon covers (kinship and register
entries for the listed languages; ``name_variant`` has no lexicon entries and
is checked by script only when a value is supplied). It is a lexicon-relative
attribution aid, not a measurement of FCR; FCR needs the gold labels.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from dkmem.memory.extract import _parse_lang_tag
from dkmem.memory.gate import DEFAULT_DISCRIMINATIVE_FEATURES, MERGING_DECISIONS, compatible
from dkmem.memory.lexicon import Lexicon
from dkmem.memory.matcher import match_distinctions
from dkmem.memory.native_extract import script_shares

__all__ = [
    "canonical_json",
    "sha256_text",
    "sha256_file",
    "sha256_file_lf",
    "pair_input_hash",
    "inputs_hash",
    "config_fingerprint",
    "git_state",
    "lexicon_distinction",
    "distinction_visibility",
    "diagnose",
    "diagnose_dropped",
    "normalize_text",
    "source_hashes",
    "LOSS_POINTS",
    "STORAGE_STATES",
]

LOSS_POINTS = ("no_detectable_difference", "dropped_at_extraction", "storage", "host_merge", "none")
STORAGE_STATES = ("visible", "collapsed", "distinct_not_visible")


def canonical_json(obj: Any) -> str:
    """Stable JSON text (sorted keys, no spaces, non-ASCII kept) for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_file_lf(path: str | Path) -> str:
    """sha256 of a text file with CRLF normalized to LF, so a checkout with ``autocrlf``
    hashes the same as one without (used for the lexicon; input data files keep the
    raw-byte hash their contracts specify)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pair_input_hash(eval_pair_id: str, language: str, utterance_a: str, utterance_b: str) -> str:
    """Hash of exactly the fields a system may read for one pair.

    The gold-passthrough entity ids are excluded: they never influence a run.
    """
    return sha256_text(
        canonical_json(
            {"eval_pair_id": eval_pair_id, "language": language, "utterance_a": utterance_a, "utterance_b": utterance_b}
        )
    )


def inputs_hash(pair_hashes: list[str]) -> str:
    """Hash of the ordered list of per-pair input hashes."""
    return sha256_text(canonical_json(pair_hashes))


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """sha256 of a run configuration (no timestamps or paths of outputs in it)."""
    return sha256_text(canonical_json(config))


_SOURCE_SUFFIXES = (".py", ".json")


def source_hashes(package_dir: str | Path | None = None) -> dict[str, Any]:
    """sha256 of every source and data file of the ``dkmem`` package, for ``run_config.json``.

    Covers ``*.py`` and ``*.json`` (schemas, the lexicon) under the package
    directory (default: the installed ``dkmem``), skipping ``__pycache__`` and the
    ``data`` directory (input data has its own hashes). Line endings are
    normalized (CRLF -> LF) so a checkout with ``autocrlf`` hashes the same as one
    without. Returns ``{"files": {relpath: sha256}, "tree_sha256": ...}``.
    """
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent.parent
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts or rel.parts[0] == "data":
            continue
        data = path.read_bytes().replace(b"\r\n", b"\n")
        files[rel.as_posix()] = hashlib.sha256(data).hexdigest()
    return {"files": files, "tree_sha256": sha256_text(canonical_json(files))}


def git_state(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """``{"commit", "dirty"}`` of the code checkout; ``None`` values if git is unavailable."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(repo_dir) if repo_dir else None,
                capture_output=True, text=True, timeout=10, check=True,
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": (None if status is None else bool(status))}


_NORM_STRIP = re.compile(r"[\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Casefolded text with every non-letter, non-digit run removed (for equality tests only)."""
    return _NORM_STRIP.sub("", text.casefold())


def lexicon_distinction(lexicon: Lexicon, text: str, lang: str) -> dict[str, str]:
    """The lexicon's distinction dict for ``text`` (same convention as ``extract_lexicon_only``)."""
    result: dict[str, str] = {}
    for component in _parse_lang_tag(lang):
        result.update(match_distinctions(lexicon, text, component))
    return result


def distinction_visibility(
    lexicon: Lexicon, distinction: Mapping[str, str], stored_text: str, lang: str
) -> dict[str, bool]:
    """For each ``key: value`` of ``distinction``, is it still visible in ``stored_text``?

    Visible means the lexicon finds the same value in the stored text; a
    ``name_variant`` value ``"<Name>-<script>"`` is visible when the stored text
    contains letters of that script.
    """
    found = lexicon_distinction(lexicon, stored_text, lang)
    scripts = script_shares(stored_text)
    out = {}
    for key, value in distinction.items():
        if key == "name_variant":
            script = value.rsplit("-", 1)[-1].upper()
            out[key] = script in scripts
        else:
            out[key] = found.get(key) == value
    return out


def diagnose(
    *,
    lexicon: Lexicon,
    lang: str,
    utterance_a: str,
    utterance_b: str,
    stored_text_a: str,
    stored_text_b: str,
    host_proposed: str,
    storage_loss_stage: str,
    host_loss_stage: str,
) -> dict[str, Any]:
    """The loss-point diagnosis of one candidate pair (see the module docstring).

    ``storage_loss_stage`` / ``host_loss_stage`` are the config's L-labels for
    the two points (``L1``/``L2`` for storage; ``L3``/``L4`` for the host).
    """
    dist_a = lexicon_distinction(lexicon, utterance_a, lang)
    dist_b = lexicon_distinction(lexicon, utterance_b, lang)
    compat = compatible(dist_a, dist_b)

    differing = {
        k for k in DEFAULT_DISCRIMINATIVE_FEATURES if dist_a.get(k) != dist_b.get(k)
    }
    vis_a = distinction_visibility(lexicon, {k: v for k, v in dist_a.items() if k in differing}, stored_text_a, lang)
    vis_b = distinction_visibility(lexicon, {k: v for k, v in dist_b.items() if k in differing}, stored_text_b, lang)
    all_visible = all(vis_a.values()) and all(vis_b.values())
    collapsed = normalize_text(stored_text_a) == normalize_text(stored_text_b)

    if compat == "compatible":
        storage_state = None
        first_loss, stage = "no_detectable_difference", None
    else:
        storage_state = "visible" if all_visible else "collapsed" if collapsed else "distinct_not_visible"
        if storage_state == "collapsed":
            first_loss, stage = "storage", storage_loss_stage
        elif host_proposed in MERGING_DECISIONS:
            first_loss, stage = "host_merge", host_loss_stage
        else:
            first_loss, stage = "none", None

    return {
        "lexicon_distinction_a": dist_a,
        "lexicon_distinction_b": dist_b,
        "lexicon_compatibility": compat,
        "differing_features": sorted(differing),
        "visible_in_stored_a": vis_a,
        "visible_in_stored_b": vis_b,
        "distinction_survives_storage": all_visible if compat != "compatible" else None,
        "storage_state": storage_state,
        "stored_texts_identical": collapsed,
        "host_proposed": host_proposed,
        "first_loss_point": first_loss,
        "loss_stage": stage,
    }


def diagnose_dropped(
    *,
    lexicon: Lexicon,
    lang: str,
    utterance_a: str,
    utterance_b: str,
    extraction_loss_stage: str | None,
    sides_dropped: list[str],
) -> dict[str, Any]:
    """Diagnosis of an episode where at least one side stored nothing.

    ``first_loss_point`` is ``dropped_at_extraction``: there is no entry on the
    dropped side(s), so no merge decision exists and none of the other loss points
    applies. ``lexicon_detectable_difference`` says whether the utterances carried
    a lexicon-visible distinction at all (so analyses can restrict to pairs where
    a drop could have lost one). ``extraction_loss_stage`` is the config's
    extraction label (``L1`` for Config A, ``L2`` for B).
    """
    dist_a = lexicon_distinction(lexicon, utterance_a, lang)
    dist_b = lexicon_distinction(lexicon, utterance_b, lang)
    compat = compatible(dist_a, dist_b)
    return {
        "lexicon_distinction_a": dist_a,
        "lexicon_distinction_b": dist_b,
        "lexicon_compatibility": compat,
        "lexicon_detectable_difference": compat != "compatible",
        "sides_dropped": sorted(sides_dropped),
        "storage_state": None,
        "host_proposed": None,
        "first_loss_point": "dropped_at_extraction",
        "loss_stage": extraction_loss_stage,
    }
