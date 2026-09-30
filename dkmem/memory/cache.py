"""Persistent cache of completed ``Extraction`` results.

Wraps ``dkmem.memory.extract`` so a given extraction is generated at most once.
Entries live in an append-only UTF-8 JSONL file, one per line:

    {"key": <sha256>, "key_fields": {...}, "extraction": {<Extraction>}}

The key is the SHA-256 of the canonical JSON of ``key_fields``: the probe
side (pair_id, side, utterance, lang), the prompt (id and text fingerprint),
the backbone, every ``GenerationParams`` field (seed included), and optional
``backend_info`` (e.g. revision, dtype). Different configurations therefore
never share an entry.

Every entry is verified when the file is loaded: its key must match its
fields, and its Extraction must equal what the extraction code rebuilds from
the stored ``raw_output``. Anything else raises ``CacheError``. An existing
entry is never overwritten; a different result for the same key raises
``CacheConflictError``.

Note: ``set_seed`` runs once per ``generate`` call, so sampled (non-greedy)
outputs also depend on batch composition, which the key does not capture.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from dkmem.backends.llm import GenerationParams
from dkmem.memory.extract import (
    ExtractionBatchError,
    ExtractionError,
    TextGenerator,
    derive_lang_profile,
    extract_many,
    parse_model_output,
    utterance_for,
)
from dkmem.memory.prompts import MEM0_EXTRACTION_V1, PromptTemplate, get_prompt
from dkmem.memory.schema import Extraction, ProbeItem, SchemaError, _ends_without_newline

__all__ = [
    "CACHE_FORMAT",
    "CacheError",
    "CacheConflictError",
    "CacheKey",
    "ExtractionCache",
    "cached_extract",
    "cached_extract_many",
]

CACHE_FORMAT = 1
_KEY_FIELD_NAMES = frozenset(
    {
        "cache_format",
        "pair_id",
        "side",
        "utterance",
        "lang",
        "prompt_id",
        "prompt_sha256",
        "backbone",
        "generation_params",
        "backend_info",
    }
)


class CacheError(ValueError):
    """A cache file or entry is invalid, corrupted, or stale."""


class CacheConflictError(CacheError):
    """A different result was offered for a key that is already cached."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class CacheKey:
    """Identity of one extraction: its ``fields`` and their SHA-256 ``digest``."""

    __slots__ = ("fields", "digest")

    def __init__(self, fields: Mapping[str, Any]) -> None:
        self.fields = dict(fields)
        self.digest = hashlib.sha256(_canonical_json(self.fields).encode("utf-8")).hexdigest()

    @classmethod
    def build(
        cls,
        probe: ProbeItem,
        side: str,
        *,
        prompt: PromptTemplate,
        backbone: str,
        params: GenerationParams,
        backend_info: Mapping[str, Any] | None = None,
    ) -> "CacheKey":
        """Key for extracting ``side`` of ``probe`` under this configuration."""
        if backend_info is not None:
            backend_info = dict(backend_info)
            try:
                _canonical_json(backend_info)
            except (TypeError, ValueError) as e:
                raise ValueError(f"backend_info must be JSON-serializable: {e}") from e
        return cls(
            {
                "cache_format": CACHE_FORMAT,
                "pair_id": probe.pair_id,
                "side": side,
                "utterance": utterance_for(probe, side),
                "lang": probe.lang,
                "prompt_id": prompt.prompt_id,
                "prompt_sha256": prompt.sha256,
                "backbone": backbone,
                "generation_params": asdict(params),
                "backend_info": backend_info,
            }
        )

    def __repr__(self) -> str:
        return f"CacheKey({self.digest[:12]}…)"


def _expected_extraction(fields: Mapping[str, Any], raw_output: str) -> Extraction:
    """Rebuild the Extraction that the extraction code produces for ``fields``."""
    prompt = get_prompt(fields["prompt_id"])
    if prompt.sha256 != fields["prompt_sha256"]:
        raise CacheError(
            f"prompt {fields['prompt_id']!r} text changed since caching "
            f"(cached {fields['prompt_sha256'][:12]}…, now {prompt.sha256[:12]}…)"
        )
    utterance = fields["utterance"]
    gloss, distinction, surface = parse_model_output(raw_output, utterance=utterance, prompt=prompt)
    return Extraction(
        pair_id=fields["pair_id"],
        side=fields["side"],
        raw_output=raw_output,
        gloss=gloss,
        distinction=distinction,
        surface=surface,
        lang_profile=derive_lang_profile(fields["lang"], utterance),
        backbone=fields["backbone"],
        prompt_id=fields["prompt_id"],
        seed=fields["generation_params"]["seed"],
    )


def _verify(key: CacheKey, extraction: Extraction) -> None:
    """Raise ``CacheError`` unless ``extraction`` is exactly what ``key`` implies."""
    try:
        expected = _expected_extraction(key.fields, extraction.raw_output)
    except (ExtractionError, SchemaError, KeyError, TypeError, ValueError) as e:
        if isinstance(e, CacheError):
            raise
        raise CacheError(f"entry {key!r} does not verify: {e}") from e
    if expected != extraction:
        diff = sorted(
            name for name, value in extraction.to_dict().items() if expected.to_dict()[name] != value
        )
        raise CacheError(f"entry {key!r} does not match its raw_output/key; differing fields: {diff}")


class ExtractionCache:
    """Append-only, verified JSONL cache of Extractions keyed by ``CacheKey``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, Extraction] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8-sig") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    digest, extraction = self._parse_line(line)
                except CacheError as e:
                    raise CacheError(f"{self.path}:{lineno}: {e}") from e
                existing = self._entries.get(digest)
                if existing is not None and existing != extraction:
                    raise CacheConflictError(
                        f"{self.path}:{lineno}: key {digest[:12]}… has conflicting cached results"
                    )
                self._entries[digest] = extraction

    @staticmethod
    def _parse_line(line: str) -> tuple[str, Extraction]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            raise CacheError(f"invalid JSON: {e}") from e
        if not isinstance(entry, dict) or set(entry) != {"key", "key_fields", "extraction"}:
            raise CacheError("entry must be an object with keys 'key', 'key_fields', 'extraction'")
        fields = entry["key_fields"]
        if not isinstance(fields, dict) or set(fields) != _KEY_FIELD_NAMES:
            raise CacheError(f"'key_fields' must have exactly {sorted(_KEY_FIELD_NAMES)}")
        if fields["cache_format"] != CACHE_FORMAT:
            raise CacheError(f"unsupported cache_format {fields['cache_format']!r}")
        try:
            key = CacheKey(fields)
        except (TypeError, ValueError) as e:
            raise CacheError(f"'key_fields' is not canonical JSON: {e}") from e
        if entry["key"] != key.digest:
            raise CacheError("'key' does not match the SHA-256 of 'key_fields'")
        try:
            extraction = Extraction.from_dict(entry["extraction"])
        except SchemaError as e:
            raise CacheError(f"invalid extraction record: {e}") from e
        _verify(key, extraction)
        return key.digest, extraction

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: CacheKey) -> bool:
        return key.digest in self._entries

    def get(self, key: CacheKey) -> Extraction | None:
        """The cached Extraction for ``key``, or ``None``."""
        return self._entries.get(key.digest)

    def put(self, key: CacheKey, extraction: Extraction) -> None:
        """Verify and persist ``extraction`` under ``key``.

        Re-putting an identical record is a no-op; a different record for an
        existing key raises ``CacheConflictError`` and nothing is written.
        """
        _verify(key, extraction)
        existing = self._entries.get(key.digest)
        if existing is not None:
            if existing != extraction:
                raise CacheConflictError(
                    f"key {key!r} ({key.fields['pair_id']}/{key.fields['side']}) is already cached "
                    "with a different result; refusing to overwrite"
                )
            return
        line = _canonical_json(
            {"key": key.digest, "key_fields": key.fields, "extraction": extraction.to_dict()}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = _ends_without_newline(str(self.path))
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            if needs_newline:
                f.write("\n")
            f.write(line + "\n")
        self._entries[key.digest] = extraction


def cached_extract_many(
    items: Sequence[tuple[ProbeItem, str]],
    generator: TextGenerator,
    cache: ExtractionCache,
    params: GenerationParams | None = None,
    *,
    prompt: PromptTemplate = MEM0_EXTRACTION_V1,
    backend_info: Mapping[str, Any] | None = None,
) -> list[Extraction]:
    """``extract_many`` that only generates cache misses and persists them.

    Duplicate misses are generated once. If some misses fail, the successful
    ones are still cached before the ``ExtractionBatchError`` is re-raised.
    """
    params = params or GenerationParams()
    items = list(items)
    keys = [
        CacheKey.build(p, s, prompt=prompt, backbone=generator.backbone, params=params,
                       backend_info=backend_info)
        for p, s in items
    ]

    to_generate: dict[str, int] = {}
    for i, key in enumerate(keys):
        if key not in cache and key.digest not in to_generate:
            to_generate[key.digest] = i

    if to_generate:
        miss_idx = list(to_generate.values())
        try:
            generated = extract_many([items[i] for i in miss_idx], generator, params, prompt=prompt)
        except ExtractionBatchError as e:
            for i, ex in zip(miss_idx, e.extractions):
                if ex is not None:
                    cache.put(keys[i], ex)
            raise
        for i, ex in zip(miss_idx, generated):
            cache.put(keys[i], ex)

    return [cache.get(k) for k in keys]  # type: ignore[misc]


def cached_extract(
    probe: ProbeItem,
    side: str,
    generator: TextGenerator,
    cache: ExtractionCache,
    params: GenerationParams | None = None,
    *,
    prompt: PromptTemplate = MEM0_EXTRACTION_V1,
    backend_info: Mapping[str, Any] | None = None,
) -> Extraction:
    """Single-item ``cached_extract_many``; raises ``ExtractionError`` on bad output."""
    try:
        return cached_extract_many(
            [(probe, side)], generator, cache, params, prompt=prompt, backend_info=backend_info
        )[0]
    except ExtractionBatchError as e:
        raise e.failures[0] from e
