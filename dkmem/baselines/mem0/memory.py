"""Mem0 OSS v1.0.11 add/update flow, re-implemented against in-memory stores.

``Mem0Memory.add`` mirrors ``mem0.memory.main.Memory.add`` -> ``_add_to_vector_store`` for ``infer=True`` (the
default) statement by statement; the upstream line each block follows is quoted in the comments and checked
against the pinned source by ``dkmem.baselines.mem0.source.verify_structure``. Prompts and helper functions are
the vendored upstream ones (``load_pinned``), not retyped.

Preserved (upstream behaviour, including the sloppy parts)
----------------------------------------------------------
* Fact extraction: ``USER_MEMORY_EXTRACTION_PROMPT`` (system) + ``"Input:\\n" + parse_messages(messages)`` (user), one
  LLM call, ``response_format={"type": "json_object"}`` requested. A response that cannot be parsed, or has no
  ``facts`` key, silently becomes *no facts* (and then nothing is stored); an LLM exception here is not caught.
* Per fact: embed it, search the user's memories with ``limit=5`` and **no score threshold**; collect
  ``{id, text}`` of every hit, de-duplicate by id keeping the last, then replace the ids by ``"0", "1", ...``
  (``temp_uuid_mapping``) so the LLM cannot hallucinate a UUID.
* One update call (``get_update_memory_messages``: ``DEFAULT_UPDATE_MEMORY_PROMPT`` + the remapped old memories
  + the new facts) is made only if there are facts. An exception, an empty response or unparseable JSON
  becomes ``{}`` -- no event is applied and nothing is reported.
* The returned ``{"memory": [...]}`` list is applied **sequentially**; each item is guarded by its own
  ``try``, so an item with a hallucinated id, an unknown event or a missing memory is logged and skipped while
  the others still apply. ``ADD`` creates a memory, ``UPDATE`` overwrites the mapped memory's text, ``DELETE``
  removes it, ``NONE`` does nothing (it only refreshes session ids when ``agent_id``/``run_id`` are given).
* Every create/update/delete writes a history row (old text, new text, event), as upstream.

Substituted (recorded in ``run_config.json``)
---------------------------------------------
* LLM: the run's Qwen generator, called with the same chat messages. ``response_format`` is passed to the
  adapter but a local Hugging Face model cannot enforce it (nothing constrains the output to JSON), and the
  update call has no system message (the chat template then adds its default one).
* Embedder: bge-m3 (dense CLS, normalised) instead of the OpenAI default; the vector store is an in-memory
  cosine search instead of Qdrant; history is a list instead of SQLite.
* Memory ids are deterministic uuid5 values and timestamps come from a logical clock, so runs are identical
  across repetitions; neither enters any decision.

Not implemented: ``infer=False``, custom prompts, agent memory, vision messages, the graph store, procedural
memory and the telemetry -- none is used by the baseline.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from dkmem.backends.embedding import cosine
from dkmem.baselines.mem0.source import PinnedMem0, load_pinned

__all__ = [
    "Mem0Error",
    "Mem0AddError",
    "Mem0LLM",
    "Mem0Embedder",
    "GeneratorLLM",
    "BatchEmbedder",
    "Hit",
    "InMemoryVectorStore",
    "HistoryLog",
    "AddResult",
    "Mem0Memory",
    "SEARCH_LIMIT",
]

SEARCH_LIMIT = 5  # upstream: vector_store.search(..., limit=5, ...) with no threshold


class Mem0Error(ValueError):
    """The input or configuration is outside what this reproduction supports."""


class Mem0AddError(RuntimeError):
    """``add`` failed the way upstream's would (an LLM exception during fact extraction, which upstream does not
    catch). ``trace`` holds everything recorded up to the failure, including the raw prompt."""

    def __init__(self, message: str, trace: dict[str, Any]) -> None:
        super().__init__(message)
        self.trace = trace


class Mem0LLM(Protocol):
    def generate_response(self, messages: list[dict[str, str]], response_format: Any = None) -> str: ...


class Mem0Embedder(Protocol):
    def embed(self, text: str, memory_action: str) -> list[float]: ...


class GeneratorLLM:
    """``generate_response`` on top of a ``TextGenerator`` (one prompt per call, like upstream's sequential flow).

    Every call is recorded in ``calls`` (messages, requested response format, raw output or error): the raw
    LLM input/output the baseline's trace needs.
    """

    def __init__(self, generator: Any, params: Any = None) -> None:
        self.generator, self.params = generator, params
        self.calls: list[dict[str, Any]] = []

    def generate_response(self, messages: list[dict[str, str]], response_format: Any = None) -> str:
        call: dict[str, Any] = {"messages": deepcopy(messages), "response_format": response_format}
        self.calls.append(call)
        try:
            outputs = self.generator.generate([messages], self.params)
            if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], str):
                raise ValueError(f"generator returned {outputs!r} for one prompt")
        except Exception as e:
            call["error"] = f"{type(e).__name__}: {e}"
            raise
        call["raw_output"] = outputs[0]
        return outputs[0]


class BatchEmbedder:
    """``embed(text, memory_action)`` on top of a ``texts -> vectors`` function such as ``BgeM3Embedder.embed``.

    Memoised per text (the action label does not change a bge-m3 embedding), which also guarantees that the same
    text always has the same vector -- the property one similarity comparator needs.
    """

    def __init__(self, embed_texts: Callable[[Sequence[str]], list[list[float]]]) -> None:
        self._embed_texts = embed_texts
        self._memo: dict[str, list[float]] = {}
        self.stats = {"calls": 0, "texts_embedded": 0}

    def embed(self, text: str, memory_action: str = "add") -> list[float]:
        self.stats["calls"] += 1
        if text not in self._memo:
            self._memo[text] = list(self._embed_texts([text])[0])
            self.stats["texts_embedded"] += 1
        return self._memo[text]


@dataclass
class Hit:
    id: str
    score: float | None
    payload: dict[str, Any]


class InMemoryVectorStore:
    """Cosine nearest-neighbour search with exact-match payload filters (the Qdrant stand-in)."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[list[float], dict[str, Any]]] = {}

    def insert(self, vectors: list[list[float]], ids: list[str], payloads: list[dict[str, Any]]) -> None:
        for vector, vid, payload in zip(vectors, ids, payloads):
            if vid in self._rows:
                raise Mem0Error(f"vector id {vid!r} already exists")
            self._rows[vid] = (list(vector), deepcopy(payload))

    def search(self, query: str, vectors: list[float], limit: int, filters: Mapping[str, Any] | None) -> list[Hit]:
        hits = [
            Hit(vid, cosine(vectors, vec), deepcopy(payload))
            for vid, (vec, payload) in self._rows.items()
            if all(payload.get(k) == v for k, v in (filters or {}).items())
        ]
        hits.sort(key=lambda h: -h.score)  # stable: equal scores keep insertion order
        return hits[:limit]

    def get(self, vector_id: str) -> Hit | None:
        row = self._rows.get(vector_id)
        return None if row is None else Hit(vector_id, None, deepcopy(row[1]))

    def update(self, vector_id: str, vector: list[float] | None, payload: dict[str, Any] | None) -> None:
        if vector_id not in self._rows:
            raise Mem0Error(f"vector id {vector_id!r} not found")
        old_vector, old_payload = self._rows[vector_id]
        self._rows[vector_id] = (
            list(vector) if vector is not None else old_vector,
            deepcopy(payload) if payload is not None else old_payload,
        )

    def delete(self, vector_id: str) -> None:
        if vector_id not in self._rows:
            raise Mem0Error(f"vector id {vector_id!r} not found")
        del self._rows[vector_id]

    def list(self, filters: Mapping[str, Any] | None = None) -> list[Hit]:
        return [
            Hit(vid, None, deepcopy(payload)) for vid, (_, payload) in self._rows.items()
            if all(payload.get(k) == v for k, v in (filters or {}).items())
        ]


class HistoryLog:
    """The ``history`` table (same columns as upstream's SQLite one), as a list of dict rows."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add_history(
        self, memory_id: str, old_memory: str | None, new_memory: str | None, event: str, *,
        created_at: str | None = None, updated_at: str | None = None, actor_id: str | None = None,
        role: str | None = None, is_deleted: int = 0,
    ) -> dict[str, Any]:
        row = {
            "id": f"h{len(self.rows)}", "memory_id": memory_id, "old_memory": old_memory, "new_memory": new_memory,
            "event": event, "created_at": created_at, "updated_at": updated_at, "is_deleted": is_deleted,
            "actor_id": actor_id, "role": role,
        }
        self.rows.append(row)
        return row

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.rows if r["memory_id"] == memory_id]


@dataclass
class AddResult:
    """What ``Mem0Memory.add`` returns: upstream's ``results`` list and the full trace of the call."""

    results: list[dict[str, Any]]
    trace: dict[str, Any]
    history_rows: list[dict[str, Any]] = field(default_factory=list)


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


class Mem0Memory:
    """One user's Mem0 memory (vector store + history) driven by an LLM and an embedder."""

    def __init__(
        self,
        llm: Mem0LLM,
        embedder: Mem0Embedder,
        *,
        namespace: str = "mem0",
        pinned: PinnedMem0 | None = None,
        search_limit: int = SEARCH_LIMIT,
    ) -> None:
        self.llm, self.embedding_model = llm, embedder
        self.pinned = pinned or load_pinned()
        self.vector_store = InMemoryVectorStore()
        self.db = HistoryLog()
        self._namespace = namespace
        self._n_ids = 0
        self._tick = 0
        if search_limit != SEARCH_LIMIT:
            raise Mem0Error(f"Mem0 v1.0.11 searches with limit={SEARCH_LIMIT}; got {search_limit}")

    # --- deterministic ids and clock ------------------------------------------------------------------

    def _new_id(self) -> str:
        self._n_ids += 1
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self._namespace}:{self._n_ids}"))

    def _now(self) -> str:
        self._tick += 1
        return (datetime(2026, 10, 7, tzinfo=timezone.utc) + timedelta(seconds=self._tick)).isoformat()

    # --- reads -----------------------------------------------------------------------------------------

    def get_all(self, user_id: str) -> list[dict[str, Any]]:
        return [{"id": h.id, "memory": h.payload.get("data", "")} for h in self.vector_store.list({"user_id": user_id})]

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        return self.db.get_history(memory_id)

    # --- add --------------------------------------------------------------------------------------------

    def add(self, messages: Any, *, user_id: str, metadata: Mapping[str, Any] | None = None, infer: bool = True) -> AddResult:
        """``Memory.add(messages, user_id=...)``; see the module docstring. ``messages`` is a string or a list of
        ``{"role", "content"}`` dicts with string contents."""
        if not infer:
            raise Mem0Error("infer=False is not part of the baseline")
        if not isinstance(user_id, str) or not user_id:
            raise Mem0Error("user_id is required")  # upstream: VALIDATION_001 (at least one session id)
        processed_metadata = deepcopy(dict(metadata)) if metadata else {}
        processed_metadata["user_id"] = user_id
        effective_filters = {"user_id": user_id}

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise Mem0Error("messages must be str, dict, or list[dict]")
        for m in messages:
            if not isinstance(m, dict) or not isinstance(m.get("content"), str) or "role" not in m:
                raise Mem0Error(f"unsupported message (only plain-text messages are reproduced): {m!r}")

        trace: dict[str, Any] = {"user_id": user_id, "messages": deepcopy(messages)}
        n_history = len(self.db.rows)
        try:
            results = self._add_to_vector_store(messages, processed_metadata, effective_filters, trace)
        except Mem0Error:
            raise
        except Exception as e:
            trace["exception"] = f"{type(e).__name__}: {e}"
            trace["history"] = _json_safe(self.db.rows[n_history:])
            raise Mem0AddError(trace["exception"], trace) from e
        trace["history"] = _json_safe(self.db.rows[n_history:])
        trace["results"] = _json_safe(results)
        return AddResult(results, trace, [dict(r) for r in self.db.rows[n_history:]])

    def _add_to_vector_store(self, messages, metadata, filters, trace) -> list[dict[str, Any]]:
        u, p = self.pinned.utils, self.pinned.prompts
        parsed_messages = u.parse_messages(messages)

        # system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)
        is_agent_memory = metadata.get("agent_id") is not None and any(m.get("role") == "assistant" for m in messages)
        if is_agent_memory:
            raise Mem0Error("agent memory extraction is not part of the baseline")
        system_prompt, user_prompt = u.get_fact_retrieval_messages(parsed_messages, is_agent_memory)
        # system_prompt, user_prompt = ensure_json_instruction(system_prompt, user_prompt)
        system_prompt, user_prompt = u.ensure_json_instruction(system_prompt, user_prompt)

        extraction: dict[str, Any] = {
            "system_prompt": system_prompt, "user_prompt": user_prompt, "response_format": {"type": "json_object"},
            "raw_output": None, "llm_error": None, "facts_error": None, "facts_path": None,
        }
        trace["extraction"] = extraction
        try:
            response = self.llm.generate_response(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:  # upstream does not catch this: the add() call fails
            extraction["llm_error"] = f"{type(e).__name__}: {e}"
            raise
        extraction["raw_output"] = response

        try:
            cleaned_response = u.remove_code_blocks(response)
            if not cleaned_response.strip():
                new_retrieved_facts = []
                extraction["facts_path"] = "empty_response"
            else:
                try:
                    # new_retrieved_facts = json.loads(cleaned_response, strict=False)["facts"]
                    new_retrieved_facts = json.loads(cleaned_response, strict=False)["facts"]
                    extraction["facts_path"] = "direct"
                except json.JSONDecodeError:
                    extracted_json = u.extract_json(response)
                    new_retrieved_facts = json.loads(extracted_json, strict=False)["facts"]
                    extraction["facts_path"] = "extract_json"
                # new_retrieved_facts = normalize_facts(new_retrieved_facts)
                new_retrieved_facts = u.normalize_facts(new_retrieved_facts)
        except Exception as e:
            extraction["facts_error"] = f"{type(e).__name__}: {e}"
            extraction["facts_path"] = "failed"
            new_retrieved_facts = []
        extraction["facts"] = list(new_retrieved_facts)

        retrieved_old_memory: list[dict[str, Any]] = []
        new_message_embeddings: dict[str, list[float]] = {}
        search_filters = {}
        if filters.get("user_id"):
            search_filters["user_id"] = filters["user_id"]
        retrieval = []
        for new_mem in new_retrieved_facts:
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            # existing_memories = self.vector_store.search(query=new_mem, vectors=..., limit=5, filters=search_filters)
            existing_memories = self.vector_store.search(
                query=new_mem, vectors=messages_embeddings, limit=SEARCH_LIMIT, filters=search_filters
            )
            retrieval.append({
                "fact": new_mem,
                "hits": [{"id": m.id, "score": m.score, "text": m.payload.get("data", "")} for m in existing_memories],
            })
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})
        trace["retrieval"] = retrieval

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item  # the last duplicate wins, the first position stays
        retrieved_old_memory = list(unique_data.values())

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)
        trace["retrieved_old_memory"] = deepcopy(retrieved_old_memory)
        trace["id_mapping"] = dict(temp_uuid_mapping)

        update: dict[str, Any] = {
            "called": False, "prompt": None, "raw_output": None, "llm_error": None, "parse_error": None,
            "parsed": None, "response_format": {"type": "json_object"},
        }
        trace["update"] = update
        if new_retrieved_facts:
            function_calling_prompt = p.get_update_memory_messages(retrieved_old_memory, new_retrieved_facts, None)
            update["called"], update["prompt"] = True, function_calling_prompt
            try:
                response = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                update["llm_error"] = f"{type(e).__name__}: {e}"
                response = ""
            update["raw_output"] = response

            try:
                if not response or not response.strip():
                    new_memories_with_actions = {}
                else:
                    try:
                        # new_memories_with_actions = json.loads(remove_code_blocks(response), strict=False)
                        new_memories_with_actions = json.loads(u.remove_code_blocks(response), strict=False)
                    except json.JSONDecodeError:
                        extracted_json = u.extract_json(response)
                        new_memories_with_actions = json.loads(extracted_json, strict=False)
            except Exception as e:
                update["parse_error"] = f"{type(e).__name__}: {e}"
                new_memories_with_actions = {}
        else:
            new_memories_with_actions = {}
        update["parsed"] = _json_safe(new_memories_with_actions)

        returned_memories: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        trace["actions"] = actions
        try:
            # for resp in new_memories_with_actions.get("memory", []):
            for i, resp in enumerate(new_memories_with_actions.get("memory", [])):
                rec: dict[str, Any] = {"index": i, "raw": _json_safe(resp), "event": None, "id": None, "mapped_id": None,
                                       "text": None, "outcome": None, "memory_id": None, "error": None}
                actions.append(rec)
                try:
                    action_text = resp.get("text")
                    rec["text"] = action_text
                    if not action_text:
                        rec["outcome"] = "skipped_empty_text"
                        continue

                    event_type = resp.get("event")
                    rec["event"], rec["id"] = event_type, resp.get("id")
                    if event_type == "ADD":
                        if action_text not in new_message_embeddings:
                            new_message_embeddings[action_text] = self.embedding_model.embed(action_text, "add")
                        memory_id = self._create_memory(
                            data=action_text, existing_embeddings=new_message_embeddings, metadata=deepcopy(metadata)
                        )
                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                        rec.update(outcome="created", memory_id=memory_id)
                    elif event_type == "UPDATE":
                        if action_text not in new_message_embeddings:
                            new_message_embeddings[action_text] = self.embedding_model.embed(action_text, "update")
                        rec["mapped_id"] = temp_uuid_mapping[resp.get("id")]
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp.get("id")], data=action_text,
                            existing_embeddings=new_message_embeddings, metadata=deepcopy(metadata),
                        )
                        returned_memories.append({
                            "id": temp_uuid_mapping[resp.get("id")], "memory": action_text, "event": event_type,
                            "previous_memory": resp.get("old_memory"),
                        })
                        rec.update(outcome="updated", memory_id=rec["mapped_id"])
                    elif event_type == "DELETE":
                        rec["mapped_id"] = temp_uuid_mapping[resp.get("id")]
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append({
                            "id": temp_uuid_mapping[resp.get("id")], "memory": action_text, "event": event_type,
                        })
                        rec.update(outcome="deleted", memory_id=rec["mapped_id"])
                    elif event_type == "NONE":
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        rec["mapped_id"] = memory_id
                        if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                            raise Mem0Error("session-id refresh on NONE is not part of the baseline")
                        rec["outcome"] = "noop"  # upstream: logger.info("NOOP for Memory.")
                    else:
                        rec["outcome"] = "ignored_unknown_event"  # upstream: no branch matches; nothing happens
                except Exception as e:  # upstream: logger.error(f"Error processing memory action: ...")
                    rec["outcome"] = "error"
                    rec["error"] = f"{type(e).__name__}: {e}"
        except Exception as e:  # upstream: logger.error(f"Error iterating new_memories_with_actions: {e}")
            trace["iteration_error"] = f"{type(e).__name__}: {e}"
        return returned_memories

    # --- memory mutations (upstream _create_memory / _update_memory / _delete_memory) ---------------------

    def _create_memory(self, data: str, existing_embeddings, metadata=None) -> str:
        if isinstance(existing_embeddings, dict) and data in existing_embeddings:
            embeddings = existing_embeddings[data]
        elif not isinstance(existing_embeddings, dict):
            embeddings = existing_embeddings
        else:
            embeddings = self.embedding_model.embed(data, "add")
        memory_id = self._new_id()
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = self._now()
        new_metadata["updated_at"] = new_metadata["created_at"]
        self.vector_store.insert(vectors=[embeddings], ids=[memory_id], payloads=[new_metadata])
        self.db.add_history(
            memory_id, None, data, "ADD", created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"), actor_id=new_metadata.get("actor_id"), role=new_metadata.get("role"),
        )
        return memory_id

    def _update_memory(self, memory_id: str, data: str, existing_embeddings, metadata=None) -> str:
        existing_memory = self.vector_store.get(vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data")
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = self._now()
        for key in ("user_id", "agent_id", "run_id"):
            if key not in new_metadata and key in existing_memory.payload:
                new_metadata[key] = existing_memory.payload[key]
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if isinstance(existing_embeddings, dict) and data in existing_embeddings:
            embeddings = existing_embeddings[data]
        elif not isinstance(existing_embeddings, dict):
            embeddings = existing_embeddings
        else:
            embeddings = self.embedding_model.embed(data, "update")
        self.vector_store.update(vector_id=memory_id, vector=embeddings, payload=new_metadata)
        self.db.add_history(
            memory_id, prev_value, data, "UPDATE", created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"], actor_id=new_metadata.get("actor_id"), role=new_metadata.get("role"),
        )
        return memory_id

    def _delete_memory(self, memory_id: str, existing_memory: Hit | None = None) -> str:
        if existing_memory is None:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found")
        prev_value = existing_memory.payload.get("data", "")
        created_at = existing_memory.payload.get("created_at")
        updated_at = self._now()
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id, prev_value, None, "DELETE", created_at=created_at, updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"), role=existing_memory.payload.get("role"), is_deleted=1,
        )
        return memory_id
