"""Hash-pinned upstream Mem0 v1.0.11 sources: verification and loading of the prompts and helpers.

What is pinned
--------------
Three upstream files at tag ``v1.0.11`` (commit ``144627c4ce5bc4db6acac17cbd158065f2b27a8d``, pyproject
version 1.0.11), vendored under ``upstream/`` with a ``.txt`` suffix so they are data, not importable code:

* ``mem0/memory/main.py``      -- the add/update flow (``Memory._add_to_vector_store``); read for
  audit, never executed. ``verify_structure`` checks that the statements the reproduction relies on are
  present and that no score threshold occurs in the add path.
* ``mem0/configs/prompts.py``  -- ``USER_MEMORY_EXTRACTION_PROMPT`` and ``DEFAULT_UPDATE_MEMORY_PROMPT`` and
  ``get_update_memory_messages``. Executed (after verification) to obtain the exact prompt texts.
* ``mem0/memory/utils.py``     -- ``get_fact_retrieval_messages``, ``ensure_json_instruction``,
  ``parse_messages``, ``normalize_facts``, ``remove_code_blocks``, ``extract_json``. Executed likewise.

Each file is pinned by the sha256 of its bytes (CRLF normalized to LF) and by its git blob sha1, which equals
the blob id GitHub reports for the file at ``v1.0.11`` (checked when the pins were created). A mismatch raises
``Mem0SourceError`` before anything is executed: ``load_pinned`` verifies first, every time.

The one thing that is not frozen upstream: the extraction prompt embeds ``datetime.now()`` at import time
("Today's date is ..."). Reproducibility requires a fixed date, so the vendored module is executed with a
frozen ``datetime`` (``FROZEN_PROMPT_DATE``, the same date ``native_b_v1`` uses). The date is recorded in the
provenance next to the prompt hashes.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

__all__ = [
    "MEM0_REPO",
    "MEM0_TAG",
    "MEM0_COMMIT",
    "FROZEN_PROMPT_DATE",
    "UPSTREAM_DIR",
    "UPSTREAM_PINS",
    "Mem0SourceError",
    "PinnedMem0",
    "verify_upstream_pins",
    "verify_structure",
    "load_pinned",
]

MEM0_REPO = "https://github.com/mem0ai/mem0"
MEM0_TAG = "v1.0.11"
MEM0_COMMIT = "144627c4ce5bc4db6acac17cbd158065f2b27a8d"
FROZEN_PROMPT_DATE = "2026-10-07"

UPSTREAM_DIR = Path(__file__).resolve().parent / "upstream"

# upstream path -> (local file, sha256 of the LF-normalized bytes, git blob sha1)
UPSTREAM_PINS = {
    "mem0/memory/main.py": (
        "main.py.txt",
        "04035505fc37dce54a98e5868b6775d3f57f40c4f070e6d6463138286e98e3e1",
        "1804e7831dddd1c0365dddac5dbef0c3b49f7269",
    ),
    "mem0/configs/prompts.py": (
        "prompts.py.txt",
        "05be41d3d0ea7e877fbde2875d146a15624435e32ca04949bc4f6304a9bd7cad",
        "b7851a5105c978f301022b796511f42de10d27ef",
    ),
    "mem0/memory/utils.py": (
        "utils.py.txt",
        "0ade2e4b68122df61003ea289bbe2c86247c5dc92e18776abcb5e9fba6dcce2f",
        "61e3863e380972118d297b541e6fcb1e975f6abb",
    ),
}

# Statements of Memory._add_to_vector_store (main.py) that dkmem.baselines.mem0.pipeline re-implements.
# Each must occur verbatim in the pinned main.py; the pipeline's comments quote the same lines.
STRUCTURAL_MARKERS = (
    "system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)",
    "system_prompt, user_prompt = ensure_json_instruction(system_prompt, user_prompt)",
    'response_format={"type": "json_object"},',
    'new_retrieved_facts = json.loads(cleaned_response, strict=False)["facts"]',
    "new_retrieved_facts = normalize_facts(new_retrieved_facts)",
    "limit=5,",
    "filters=search_filters,",
    'unique_data[item["id"]] = item',
    'temp_uuid_mapping[str(idx)] = item["id"]',
    'retrieved_old_memory[idx]["id"] = str(idx)',
    "function_calling_prompt = get_update_memory_messages(",
    "retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt",
    "new_memories_with_actions = json.loads(remove_code_blocks(response), strict=False)",
    'for resp in new_memories_with_actions.get("memory", []):',
    'if event_type == "ADD":',
    'elif event_type == "UPDATE":',
    'elif event_type == "DELETE":',
    'elif event_type == "NONE":',
    'memory_id=temp_uuid_mapping[resp.get("id")],',
    'self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])',
    'logger.info("NOOP for Memory.")',
    'logger.error(f"Error processing memory action: {resp}, Error: {e}")',
)


class Mem0SourceError(RuntimeError):
    """A vendored upstream file does not match its pin, or does not have the expected structure."""


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_upstream_pins(directory: str | Path = UPSTREAM_DIR) -> dict[str, dict[str, str]]:
    """Check every vendored file against its pins; returns ``{upstream_path: {sha256, git_blob_sha1}}``.

    Raises ``Mem0SourceError`` naming the file and both values on any mismatch or missing file.
    """
    directory = Path(directory)
    out: dict[str, dict[str, str]] = {}
    for upstream_path, (local, sha256, blob) in UPSTREAM_PINS.items():
        path = directory / local
        try:
            data = _lf(path.read_bytes())
        except OSError as e:
            raise Mem0SourceError(f"{upstream_path}: cannot read vendored copy {path}: {e}") from e
        got_sha, got_blob = hashlib.sha256(data).hexdigest(), _blob_sha1(data)
        if got_sha != sha256 or got_blob != blob:
            raise Mem0SourceError(
                f"{upstream_path}: vendored copy {path} does not match the {MEM0_TAG} pin "
                f"(sha256 {got_sha} vs {sha256}; git blob {got_blob} vs {blob})"
            )
        out[upstream_path] = {"sha256": got_sha, "git_blob_sha1": got_blob, "vendored_as": local}
    return out


def verify_structure(directory: str | Path = UPSTREAM_DIR) -> None:
    """Check that the statements the reproduction mirrors are in the pinned ``main.py``, and that the add path
    has no score threshold (Mem0 v1.0.11 retrieves ``limit=5`` with no cutoff)."""
    verify_upstream_pins(directory)
    text = _lf((Path(directory) / UPSTREAM_PINS["mem0/memory/main.py"][0]).read_bytes()).decode("utf-8")
    missing = [m for m in STRUCTURAL_MARKERS if m not in text]
    if missing:
        raise Mem0SourceError(f"pinned main.py lacks expected statements: {missing}")
    start = text.index("    def _add_to_vector_store(self, messages, metadata, filters, infer):")
    end = text.index("    def _add_to_graph(self, messages, filters):", start)
    if "threshold" in text[start:end]:
        raise Mem0SourceError("the pinned add path mentions a threshold; the reproduction assumes none")


def _frozen_datetime(date: str):
    class _Now:
        def strftime(self, fmt: str) -> str:
            if fmt != "%Y-%m-%d":
                raise Mem0SourceError(f"the pinned prompts only format dates as %Y-%m-%d, got {fmt!r}")
            return date

    class _FrozenDatetime:
        @staticmethod
        def now(tz=None):
            return _Now()

    return _FrozenDatetime


def _exec_pinned(path: Path, namespace: dict[str, Any], drop_import: Callable[[ast.stmt], bool]) -> dict[str, Any]:
    """Execute a vendored upstream module body without the imports ``drop_import`` selects."""
    source = _lf(path.read_bytes()).decode("utf-8")
    tree = ast.parse(source, filename=str(path))
    tree.body = [node for node in tree.body if not drop_import(node)]
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 -- hash-verified vendored upstream code
    return namespace


def _is_import_from(prefix: str):
    return lambda node: isinstance(node, ast.ImportFrom) and (node.module or "").startswith(prefix)


@dataclass(frozen=True)
class PinnedMem0:
    """The exact upstream prompt texts and helper functions, plus where they came from."""

    prompts: SimpleNamespace
    utils: SimpleNamespace
    provenance: dict[str, Any]

    @property
    def fact_extraction_prompt(self) -> str:
        return self.prompts.USER_MEMORY_EXTRACTION_PROMPT

    @property
    def update_memory_prompt(self) -> str:
        return self.prompts.DEFAULT_UPDATE_MEMORY_PROMPT


@lru_cache(maxsize=4)
def load_pinned(date: str = FROZEN_PROMPT_DATE) -> PinnedMem0:
    """Verify the pins, then load the prompts (with ``date`` frozen) and helpers from the vendored files."""
    pins = verify_upstream_pins()
    directory = UPSTREAM_DIR

    p_ns: dict[str, Any] = {"__name__": "mem0_pinned_prompts", "datetime": _frozen_datetime(date)}
    _exec_pinned(directory / UPSTREAM_PINS["mem0/configs/prompts.py"][0], p_ns, _is_import_from("datetime"))

    u_ns: dict[str, Any] = {
        "__name__": "mem0_pinned_utils",
        "AGENT_MEMORY_EXTRACTION_PROMPT": p_ns["AGENT_MEMORY_EXTRACTION_PROMPT"],
        "FACT_RETRIEVAL_PROMPT": p_ns["FACT_RETRIEVAL_PROMPT"],
        "USER_MEMORY_EXTRACTION_PROMPT": p_ns["USER_MEMORY_EXTRACTION_PROMPT"],
    }
    _exec_pinned(directory / UPSTREAM_PINS["mem0/memory/utils.py"][0], u_ns, _is_import_from("mem0"))

    prompts = SimpleNamespace(
        USER_MEMORY_EXTRACTION_PROMPT=p_ns["USER_MEMORY_EXTRACTION_PROMPT"],
        DEFAULT_UPDATE_MEMORY_PROMPT=p_ns["DEFAULT_UPDATE_MEMORY_PROMPT"],
        get_update_memory_messages=p_ns["get_update_memory_messages"],
    )
    names = (
        "get_fact_retrieval_messages", "ensure_json_instruction", "parse_messages",
        "normalize_facts", "remove_code_blocks", "extract_json",
    )
    utils = SimpleNamespace(**{n: u_ns[n] for n in names})
    if f"Today's date is {date}." not in prompts.USER_MEMORY_EXTRACTION_PROMPT:
        raise Mem0SourceError("the frozen date did not reach the extraction prompt")

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    provenance = {
        "package": "mem0ai", "version": "1.0.11", "repo": MEM0_REPO, "tag": MEM0_TAG, "commit": MEM0_COMMIT,
        "files": pins,
        "frozen_prompt_date": date,
        "prompts": {
            "fact_extraction": {"name": "USER_MEMORY_EXTRACTION_PROMPT", "sha256": sha(prompts.USER_MEMORY_EXTRACTION_PROMPT)},
            "update_memory": {"name": "DEFAULT_UPDATE_MEMORY_PROMPT", "sha256": sha(prompts.DEFAULT_UPDATE_MEMORY_PROMPT)},
        },
    }
    return PinnedMem0(prompts, utils, provenance)
