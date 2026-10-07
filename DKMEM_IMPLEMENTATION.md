# DK-Mem: Canonical Implementation Reference

This document describes exactly what is implemented in this repository today, file by file and function by function. It is grounded entirely in the code under `dkmem/` and the real experiment outputs under `tier1_results/` — not in the research plan. Anywhere the research plan describes something that does **not** exist in code yet, this document says so explicitly under "Not implemented".

> **Revision (scope alignment, 2026-10-07).** The research direction changed to `DKMEM_NEW_RESEARCH_IDEA.md` (stage attribution over configurations A–D). The existing code was aligned to it *without* building the A–D system: distinction scope narrowed to kinship / register / name_variant (`dkmem/memory/scope.py`); a final extraction prompt `mem0_extraction_v4` added (v1–v3 kept) and made the default; `apply_gate()` added so the gate can veto a host system's proposed merge; similarity made pluggable and separated from the gate (`dkmem/memory/consolidation.py` now holds the glue); A–D / DK-Mem ON-OFF vocabulary added (`dkmem/config.py`); the Tier 1 contracts (JSON schemas, class mapping, manifest) updated. The six runs in `tier1_results/` predate this change and were produced under the old scope (see §14).

---

## 1. One-paragraph overview

DK-Mem tests one idea: when a multilingual utterance is normalized into an English "gloss" for memory storage, a source-language distinction (e.g. Hindi *chachi* vs *mausi*, both "aunt" in English) can get lost, causing two genuinely different facts/entities to look identical and get wrongly merged. The fix implemented here is to extract a second field, `distinction`, alongside the English gloss, and gate every merge decision on both the gloss similarity **and** whether the two entries' `distinction` fields actually agree.

## 2. End-to-end pipeline diagram

```
                      ┌─────────────────────┐
  one utterance  ───► │   EXTRACTION STAGE   │
  (side "a" or "b")   │  (lexicon and/or     │
                      │   Qwen2.5-3B-Instruct)│
                      └──────────┬───────────┘
                                 │
                                 ▼
                      Extraction{ gloss, distinction,
                                  surface, lang_profile,
                                  backbone, prompt_id, seed }
                                 │
                   (one Extraction per side, "a" and "b")
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │  SIMILARITY STAGE    │
                      │  gloss_similarity()  │   sim = difflib ratio
                      │  (similarity.py)     │   on normalized glosses
                      └──────────┬───────────┘
                                 │  sim (float, 0..1)
                                 ▼
                      ┌─────────────────────┐
                      │   GATE STAGE         │
                      │   gate() (gate.py)   │
                      │  compares distinction│
                      │  dicts, feature by   │
                      │  feature             │
                      └──────────┬───────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
        incompatible      underdetermined        compatible
              │                  │               (sim decides)
              ▼                  ▼              ┌────┴────┐
         keep_both        link_unresolved        ▼         ▼
                                              merge     keep_both
                                 │
                                 ▼
                      MergeEvent / PairwiseEvalRecord
                      (written to disk as JSON)
```

There is **no live, queryable "memory store"** anywhere in this pipeline (see §13). Every run — whether a unit test, a `MergeEvent` comparison, or a full Tier 1 episode — is a single, self-contained extraction-then-gate computation on exactly the entries just produced. Nothing is retrieved from a prior episode.

---

## 3. Core data contracts — `dkmem/memory/schema.py`

Three frozen `dataclasses`, each with strict `__post_init__` validation (wrong type or shape raises `SchemaError` immediately — nothing is silently coerced except numeric widening to `float`).

### `ProbeItem`
A hand-authored (or synthetic-fixture) minimal-pair test case: two utterances (`utt_a`, `utt_b`), a language tag, the gold label `gold_same_entity`, and which `distinction_class` is supposed to separate them. Used only in the research fixtures/tests, **not** in the real Tier 1 run (Tier 1 uses `Tier1Record` instead, see §12).

### `Extraction` — the actual stored memory entry
```python
Extraction(
    pair_id: str, side: "a"|"b",
    raw_output: str,            # Qwen's verbatim text, or "" if no LLM ran
    gloss: str,                 # normalized English retrieval key
    distinction: dict[str,str], # e.g. {"kinship": "chachi"}
    surface: str,                # verbatim original utterance
    lang_profile: dict[str,float], # e.g. {"hi": 0.8, "en": 0.2, "script_mix": 0.1}
    backbone: str, prompt_id: str, seed: int,
)
```
This is the single object every later stage (similarity, gate, Tier 1 I/O) consumes. It is immutable (`frozen=True`); `dkmem_extract.py` builds a *new* `Extraction` via `dataclasses.replace` rather than mutating one.

### `MergeEvent` — one consolidation decision
```python
MergeEvent(pair_id, policy, tau, sim, decision, reason, backbone, seed, prompt_id)
```
`decision` must be one of `("merge", "supersede", "keep_both", "link_unresolved")` — this is the single source of truth for the internal decision vocabulary (`DECISIONS`). **`supersede` is defined here but never produced by any code in this repo.**

### JSONL helpers
`write_jsonl` / `read_jsonl` serialize any of the above, one JSON object per line, UTF-8, LF-terminated. `write_jsonl(..., append=True)` safely resumes a log without corrupting a truncated last line.

---

## 4. Frozen prompts — `dkmem/memory/prompts.py`

Four versioned prompt templates (`mem0_extraction_v1`, `_v2`, `_v3`, `_v4`), each a `PromptTemplate(prompt_id, system, user_template, output_keys, examples)`. A prompt's text must never change once an experiment has logged its `prompt_id` — `PromptTemplate.sha256` fingerprints the exact text so a change would be detected by the cache (§10). `DEFAULT_EXTRACTION_PROMPT` (currently **v4**) is what `extract`, `cached_extract` and `dkmem_extract` use when no prompt is passed.

All three instruct Qwen to return **exactly**:
```json
{"gloss": "...", "distinction": {...}, "surface": "..."}
```
- `gloss`: one third-person English sentence, generic nouns only (kinship collapses to "uncle"/"aunt" etc.), no non-English words, no trailing period.
- `distinction`: an object using only the fixed keys the prompt allows, populated only when the utterance actually marks that feature. v1–v3 allow 7 keys (`kinship`, `register`, `politeness`, `evidentiality`, `classifier`, `temporal_deixis`, `name_variant`); **v4 allows only the 3 in-scope keys** (`kinship`, `register`, `name_variant`).
- `surface`: the utterance copied back verbatim (used as an integrity check, see §6).

v1 → v3 differ in how much the system prompt explains and exemplifies each distinction key (v2 has 13 inline examples; v3 has the same 13 as separate user/assistant chat turns plus extra negative rules). In an earlier 20-item dev-set check on Qwen2.5-3B (greedy), target-feature distinction accuracy was v1 7/20, v2 10/20, v3 10/20 (exact-match 7/9/6 of 20; those items were also used to diagnose failures, so the numbers are optimistic).

**v4 (final, current default)** is v3 restricted to the in-scope classes: the politeness, evidentiality, classifier and temporal_deixis definitions are removed, together with the examples that only demonstrated them (the two Turkish examples stay and tag only the name). One sentence is added ("Use only the keys below", as in v1) because the parser rejects any other key. v3 was chosen as the base because on the 12 in-scope probe sides of that check v2 and v3 were tied (7/12), v3's extra spurious keys were mostly the now-removed `politeness`, and v3 had no "you"/"My …" gloss violations (v2: 4). **v4 has not been run on a model yet**; because its text differs from v3, the 10/20 figure does not carry over. v1–v3 are unchanged and kept for comparison; their fingerprints are frozen in `tests/test_prompts.py`.

`PromptTemplate.render(utterance)` returns chat messages (`[{"role": "system", ...}, {"role": "user", ...}]`) — no generation or parsing happens in this file.

---

## 5. Qwen backend — `dkmem/backends/llm.py`

`HFGenerator` wraps a Hugging Face causal LM. No prompt design or output parsing lives here — purely "take prompts in, get raw text out."

- **`ModelConfig`**: `model_id` (default `"Qwen/Qwen2.5-3B-Instruct"`), `revision`, `device` (default `"cuda:0"`), `dtype` (default `"float16"`, fp16 because the target hardware is a Kaggle T4 with no native bf16), `use_chat_template`.
- **`GenerationParams`**: `max_new_tokens=256`, `do_sample=False` (greedy by default), `temperature`/`top_p`/`top_k`/`repetition_penalty`, `seed=0`, `batch_size=8`.
- **`HFGenerator.load(config)`**: loads tokenizer + model, forces left-padding (required for decoder-only batched generation), and **overwrites the checkpoint's shipped generation defaults** (Qwen2.5-Instruct ships `temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05`) with neutral ones — so only `GenerationParams` controls decoding, never the checkpoint's own defaults.
- **`HFGenerator.generate(prompts, params)`**: batches prompts (`batched()`), calls `set_seed(params.seed)` once per call, runs `model.generate`, strips the prompt and special tokens off the output, returns one raw string per prompt in input order. No JSON parsing happens here — that's `extract.py`'s job.
- **`run_info()`**: provenance dict (backbone, model config, torch/transformers versions, chat-template hash) — used for cache keys and audit, not for decisions.

---

## 6. Baseline ("Mem0-style") extraction — `dkmem/memory/extract.py`

This is the unmodified, frozen extractor that both DK-Mem variants build on.

**`extract_many(items, generator, params, prompt)`**: for a batch of `(ProbeItem, side)` pairs, renders each into a prompt, calls `generator.generate(...)` once for the whole batch, then validates and parses every raw output via `parse_model_output`.

**`parse_model_output(raw_output, utterance, prompt)`** — strict validation, nothing is auto-repaired:
1. Must be exactly one JSON object (`json.loads` with `object_pairs_hook` that rejects duplicate keys) — code fences, extra text, or multiple objects all raise `ExtractionError`.
2. Must have exactly the prompt's `output_keys` — no missing, no extra.
3. `gloss` must be a non-empty string.
4. `distinction` must be a dict using only the keys registered for the prompt in use (`DISTINCTION_KEYS[prompt_id]`: the 7 legacy classes for v1–v3, the 3 in-scope classes for v4 — a stray key such as `politeness` under v4 raises `ExtractionError`); for the 3 closed-vocabulary legacy keys (`politeness`, `evidentiality`, `temporal_deixis`), the value must be one of a fixed small set (`CLOSED_DISTINCTION_VALUES`).
5. `surface` **must equal the input utterance exactly, character for character.** This is the integrity check that catches a model paraphrasing or truncating the original text instead of copying it.

Any failure raises `ExtractionError` carrying the raw output (nothing generated is ever silently dropped); a batch failure raises `ExtractionBatchError` holding every per-item failure plus the successful extractions.

**`derive_lang_profile(lang, utterance)`** — a fully deterministic function, **never produced by the model**:
- Single-language tag (e.g. `"hi"`) → `{"hi": 1.0, "en": 0.0}`.
- Code-mixed tag (e.g. `"hi-en"`) → the share of Latin-script tokens that are in a small hardcoded English word list (`_ENGLISH_WORDS`), the rest attributed to the other language.
- `script_mix`: share of tokens not in the utterance's majority script.

**`build_extraction(...)`** assembles the final `Extraction`, combining the parsed `(gloss, distinction, surface)` with the deterministic `lang_profile` and the caller-supplied `backbone`/`seed`/`prompt_id`.

---

## 7. The distinction lexicon — `dkmem/memory/lexicon.py` + `distinction_features.json`

A static, hand-curated JSON table (schema version `"1.0"`) mapping source-language surface markers to `Extraction.distinction` values. **Currently 15 entries total**, all pilot-reviewed, all Hindi:

| Language | Distinction class | Count | Values |
|---|---|---|---|
| `hi` (Hindi) | `kinship` | 12 | tau, chacha, tai, chachi, mama, mausa, mami, mausi, nana, dadi, nani, bua |
| `hi` (Hindi) | `register` | 3 | tu, tum, aap |

The Turkish `-mış/-miş/-muş/-müş` evidentiality entry was **removed** with the scope change. `evidentiality`, `politeness`, `classifier` and `temporal_deixis` are out of scope (`dkmem/memory/scope.py`); `name_variant` is in scope but has **no entries** (an open vocabulary; whether and how a dictionary covers it is undecided). The Kinbank-generated kinship slice, honorific/register entries beyond Hindi, and a non-Indic kinship slice are still to be built. The loader accepts any snake_case class; `tests/test_lexicon.py` checks the shipped file stays within the in-scope classes.

Each `LexiconEntry` has a `match_type` of `exact_word`, `prefix`, `suffix`, or `regex`, a tuple of `surface_forms`, and a `value`. `load_lexicon()` validates every entry and rejects: duplicate IDs, and two entries that map the *same* surface form (for the same lang/class/match_type) to *different* values. For closed-vocabulary classes, an entry's `value` must already be in `CLOSED_DISTINCTION_VALUES` — a lexicon entry can never produce a value the extraction schema would reject downstream.

---

## 8. The matcher — `dkmem/memory/matcher.py`

Pure deterministic text scanning — no model, no `Extraction` objects, no dependency on `extract.py`.

1. **Tokenize** (`_tokenize`): split on whitespace, trim only Unicode punctuation/symbol characters off each token's edges (deliberately *not* `\W`, since `\W` would also strip Devanagari vowel signs that are genuinely part of the word).
2. **Match** (`find_matches`): for one language code, scan every token against every lexicon entry for that language; `exact_word`/`prefix`/`suffix` compare case-insensitively (`casefold`), `regex` entries use `re.search` as-is.
3. **Resolve conflicts** (`match_distinctions`): if two matches disagree on the value for the *same* distinction class, break the tie deterministically: higher `priority` wins → earlier token position wins → longer matched span wins → lexicographically lower entry `id` wins. Two matches that *agree* are not a conflict.

Output: `dict[str, str]`, exactly the shape `Extraction.distinction` expects.

---

## 9. The lexicon-first, model-second cascade — `dkmem/memory/dkmem_extract.py`

This is "DK-Mem's" actual extraction strategy (as opposed to the raw baseline in §6):

1. Run the unmodified baseline extraction (`extract_many`, cached or not) to get a normal `Extraction`, including the model's own guess at `distinction`.
2. Scan the utterance with the matcher (§8), once per `-`-separated language component (so `"hi-en"` checks both `hi` and `en` entries).
3. Per distinction class:
   - Lexicon found **exactly one** candidate value → **override** the model's value (lexicon wins).
   - Lexicon found **multiple conflicting** candidate values (`_ambiguous_classes`) → **set the lexicon result aside, keep the model's value** instead.
   - Lexicon found **nothing** → keep the model's value unchanged.
4. Return a new `Extraction` (via `dataclasses.replace`) with `distinction` replaced by the merge above and `prompt_id` suffixed with `+dkmem_lexicon` (`dkmem_prompt_id()`) — so a lexicon-augmented record is never mistaken for, or cache-collides with, a pure baseline record.

`dkmem_extract_many` / `dkmem_extract` are the batch/single-item entry points; this is what `tier1/runner.py`'s `dk-mem-lexicon-llm` strategy calls.

---

## 10. Persistent extraction cache — `dkmem/memory/cache.py`

**This is the only thing in the codebase that persists an `Extraction` across calls — and it is explicitly a cache, not a memory store** (nothing ever queries it by meaning; it is keyed purely by configuration identity).

- **Format**: append-only UTF-8 JSONL, one line per entry: `{"key": <sha256 hex>, "key_fields": {...}, "extraction": {...}}`.
- **Key**: SHA-256 of the canonical JSON of: `pair_id`, `side`, `utterance`, `lang`, `prompt_id` + its text fingerprint, `backbone`, every `GenerationParams` field (seed included), and optional `backend_info`. Two different configurations (different seed, different prompt version, different model) never collide.
- **Verification on every load and every `put`**: an entry's stored `Extraction` must exactly equal what re-parsing its own `raw_output` through the current extraction code produces (`_verify`/`_expected_extraction`). A mismatch raises `CacheError` — the cache can never silently serve a stale or hand-edited result.
- **Write-once semantics**: `put()` on an existing key with a *different* result raises `CacheConflictError` and writes nothing; re-putting an identical result is a no-op.
- `cached_extract_many`/`cached_extract` wrap `extract_many`/`extract`, generating only the cache misses and persisting them before returning.

**Not used anywhere in the real Tier 1 run** (`tier1/runner.py` explicitly does not pass a `cache`) — Tier 1 episodes are deliberately independent with zero carried-over state (see §12, §13).

---

## 11. Similarity — `dkmem/memory/similarity.py` and `consolidation.py`

`similarity.py` owns only candidate pairs and text similarity and **imports nothing from the gate**. The glue that feeds a similarity score into the gate is `evaluate_pair` in `dkmem/memory/consolidation.py` (moved there from `similarity.py`).

### The formula

```python
def normalize_gloss(gloss):
    return re.sub(r"\s+", " ", gloss).strip().casefold()

def gloss_similarity(gloss_a, gloss_b):
    x, y = sorted((normalize_gloss(gloss_a), normalize_gloss(gloss_b)))
    return difflib.SequenceMatcher(None, x, y).ratio()
```

- `normalize_gloss` only casefolds and collapses whitespace — purely textual, no semantics.
- The two normalized strings are **sorted** before comparing, solely to make the result independent of call order (`SequenceMatcher.ratio()` is not perfectly symmetric on its own).
- `SequenceMatcher.ratio()` implements the **Ratcliff/Obershelp algorithm**:

  $$\text{ratio} = \frac{2M}{T}$$

  where `T = len(x) + len(y)` and `M` is the total number of matching characters found by recursively locating the single longest contiguous matching block between the two strings, then recursing on the text before and after that block, summing all matched-block lengths. It is a **character-sequence** metric — no tokenizer, no embeddings, no word-level meaning at all.

This is explicitly documented (module docstring) as a deterministic, free, reproducible **stand-in** for a real embedding similarity (e.g. bge-m3) — not the paper's intended final metric. No embedding model is called anywhere in this repository.

### Pluggable metrics: `SimilarityMetric`, `entry_similarity`
```python
SimilarityMetric(name: str, fn: Callable[[str, str], float])   # frozen dataclass
DIFFLIB_RATIO = SimilarityMetric("difflib_ratio_v1", gloss_similarity)
DEFAULT_SIMILARITY = DIFFLIB_RATIO
entry_similarity(entry_a, entry_b, metric=DEFAULT_SIMILARITY, text_field="gloss")  # "gloss" | "surface"
```
- Calling a `SimilarityMetric` checks the score is finite and within [0, 1] (the range `pairwise_eval.schema.json` requires) and **raises `SimilarityError` instead of clipping**: the reported score must stay raw, so a metric like cosine similarity (range [-1, 1]) has to map itself into [0, 1] explicitly.
- `text_field` selects which `Extraction` field is compared: the gloss by default, the verbatim surface for configurations that compare surface text (C/D in `dkmem/config.py`).
- Replacing the placeholder with a real embedding is therefore: wrap its `(text_a, text_b) -> float` in a `SimilarityMetric` and pass it as `similarity=` to `evaluate_pair` or the Tier 1 runner. Neither the gate nor any caller changes. (A real τ also has to be chosen for that metric: `DEFAULT_TAU = 0.85` belongs to the difflib ratio.) No embedding metric exists yet.

### `candidate_pairs(entries_a, entries_b)`
The cross product of two entry lists (all of `entries_a` must be `side="a"`, all of `entries_b` must be `side="b"`, all must share one `pair_id`). This is **not retrieval** — it does not search any store; it assumes the two entry lists for one episode are already known. Today, every extractor produces exactly one entry per side, so this is always a single pair.

### `consolidation.evaluate_pair(entry_a, entry_b, tau, *, policy, discriminative_features, similarity, text_field)`
Computes `sim = entry_similarity(...)` once, passes it unchanged into `gate.build_merge_event(...)` (the standalone `gate()` policy, §12), and returns the resulting `MergeEvent`. `sim` is never clipped, rounded, or overwritten after this point. The two entries must share backbone, prompt_id, seed and pair_id.

---

## 12. The compatibility gate — `dkmem/memory/gate.py`

This is the paper's actual mechanism: replacing `merge(a,b) if sim > tau` with `merge(a,b) if sim > tau AND compatible(distinction_a, distinction_b)`.

### `DEFAULT_DISCRIMINATIVE_FEATURES`
```python
{"kinship", "register"}      # = dkmem.memory.scope.DISCRIMINATIVE_CLASSES
```
These are the in-scope cannot-link classes. `name_variant` is a recognized in-scope `distinction` key but is **not** in this set, so it never participates in gating: Sec 6.1 of the research plan includes same-entity name variants (Priya / प्रिया) that should merge, so a differing name variant must not block a merge. Whether name variants should constrain merging at all is undecided. The pre-rescope six-feature set is kept as `LEGACY_DISCRIMINATIVE_CLASSES` in `dkmem/memory/scope.py` (used by the frozen fixtures and as an ablation override).

### `compatible()` / `_compatibility_detail()` — pure dict comparison, zero model/embedding calls
Loop over the discriminative features **in sorted order**; for each, look at `a = distinction_a.get(feature)`, `b = distinction_b.get(feature)`:

| Both sides | Result for this feature |
|---|---|
| Same value (or both absent) | nothing — no finding |
| Different values | candidate **incompatible** |
| One present, one absent | candidate **underdetermined** |

The function keeps the *first* incompatible finding and the *first* underdetermined finding it encounters (in sorted-feature order), then:
- Any incompatible finding exists → overall `"incompatible"` (wins even over an underdetermined finding on a different feature).
- Else any underdetermined finding exists → `"underdetermined"`.
- Else → `"compatible"`.

### `gate(distinction_a, distinction_b, sim, tau, discriminative_features)` → `(compatibility, decision, reason)`

| Compatibility | Decision | Does τ matter? |
|---|---|---|
| `incompatible` | `keep_both` | **No** — always, regardless of `sim` |
| `underdetermined` | `link_unresolved` | **No** — always, regardless of `sim` |
| `compatible` | `merge` if `sim >= tau`, else `keep_both` | **Yes** — the only branch where similarity is consulted |

**τ (tau) is a plain caller-supplied float, not a constant inside `gate.py`.** In the real Tier 1 runs it is `DEFAULT_TAU = 0.85` (`dkmem/tier1/runner.py`). It is recorded unchanged on every output record (`MergeEvent.tau` / `PairwiseEvalRecord.threshold`).

`build_merge_event(...)` wraps `gate()` into a `MergeEvent`, encoding the compatibility label as the leading word of `reason` (e.g. `"incompatible kinship values: chachi vs mausi"`).

### `apply_gate(proposed, distinction_a, distinction_b, discriminative_features)` → `GateResult` — the veto layer
`gate()` derives its own merge proposal from `sim >= tau`; it cannot sit in front of another system's decision. `apply_gate` can: the *host* mechanism (an LLM judge or an embedding threshold, as in configurations A–D) decides first, and the gate only intervenes when the host proposed a unifying decision (`merge` or `supersede`):

| Host proposed | Compatibility | Final decision |
|---|---|---|
| `merge` / `supersede` | `incompatible` | `keep_both` (vetoed) |
| `merge` / `supersede` | `underdetermined` | `link_unresolved` (vetoed) |
| `merge` / `supersede` | `compatible` | unchanged |
| `keep_both` / `link_unresolved` | any | unchanged (nothing to veto) |

`GateResult(compatibility, proposed, decision, reason)` has a `vetoed` property (`decision != proposed`); `compatibility` is reported even when nothing is vetoed. Reasons read e.g. `vetoed supersede: incompatible kinship values: chachi vs mausi`. For a host that would merge, `apply_gate("merge", …)` and `gate(…, sim >= tau)` give identical results.

**One deliberate difference from `gate()`:** `gate()` links an underdetermined pair *regardless of* `sim` (even far below τ), while `apply_gate` never links a pair the host would not have merged. This matters for the unresolved-link (bloat) count: in the six existing runs, 38 of 138 LLM rows and 10 of 173 lexicon-only rows are underdetermined, and under veto-only semantics only those at or above τ would be links (7 of the 38 LLM rows and 7 of the 10 lexicon-only rows; all 10 lexicon-only rows are the Turkish evidential pairs, §14). The Tier 1 runner still uses `gate()`; whether it should switch is an open decision.

**Neither function produces `supersede` by itself** — recognizing an "update" (e.g. a changed address) rather than a duplicate needs a rule this codebase does not implement. `apply_gate` can only veto a `supersede` that a host proposed.

---

## 13. Tier 1 evaluation pipeline — `dkmem/tier1/`

This is where the above pieces are actually run against real data, producing the files Team B scores externally.

### Input — `tier1/io.py::load_tier1_input`
Reads **only** `data/tier1/eval_input/teamA_tier1_v2.jsonl` (173 records), and refuses to proceed unless its SHA-256 matches a hardcoded expected hash (`TIER1_INPUT_SHA256`) and it has exactly 173 lines. Each line is a `Tier1Record`: `eval_pair_id`, `language`, `utterance_a`, `utterance_b`, `utterance_a_id`, `utterance_b_id`, `opaque_entity_id_a`, `opaque_entity_id_b`. The two `opaque_entity_id_*` fields are **passthrough only** — copied straight to the output's `gold_entity_id`, never read by extraction, the lexicon, or the gate.

### Two strategies — `tier1/extraction.py` + `tier1/runner.py`

| Strategy | Function | LLM call? | `backbone` recorded |
|---|---|---|---|
| `dk-mem-lexicon` | `extract_lexicon_only` | No | `null` |
| `dk-mem-lexicon-llm` | `dkmem_extract_many` (§9), default prompt v4 | Yes, Qwen2.5-3B-Instruct | model id |

`extract_lexicon_only` (no-LLM strategy): `distinction` comes purely from `match_distinctions`; since there is no translation step, `gloss` is explicitly set to the **raw surface utterance, unchanged** (not a real English gloss) — the module docstring states this is reported honestly, not disguised.

### Per-episode independence
Both `run_episode_*` functions take exactly one `Tier1Record` and return a fresh result with **no shared mutable state, cache, or carried-over object** between episodes — the only shared object is the read-only `Lexicon`. If extraction fails for either side (bad model output, unsupported language tag), that episode simply yields **no entries, not a crash** — matching the eval contract's "if an utterance yields no entries, emit nothing; Team B scores that pair as 'no entry'".

### Translating to the external contract — `tier1/io.py`
`dkmem.memory.schema.DECISIONS` → `pairwise_eval.schema.json`'s vocabulary:

| Internal (`gate.py`) | External (`pairwise_eval.jsonl`) |
|---|---|
| `merge` | `merge` |
| `keep_both` | `no_merge` |
| `link_unresolved` | `underdetermined_link` |
| *(never produced)* | `supersede` |

Internal distinction keys map onto the external schema's class enum, which now holds only the three in-scope classes: `kinship`→`kinship`, `register`→`honorific_register`, `name_variant`→`name_variant`. An out-of-scope key (e.g. from a legacy prompt) has no external name and raises `Tier1InputError`. **If an entry's internal `distinction` dict has more than one key, only the alphabetically-first one is reported externally** — a serialization limitation; the gate itself still considers every key when deciding merge/no_merge.

`write_run_manifest` / `write_pairwise_eval` produce the two files per run, validated against `run_manifest.schema.json` / `pairwise_eval.schema.json`. The manifest can now also record two optional fields: `pipeline_config` (`"A"`–`"D"`) and `dkmem_mode` (`"off"`, `"lexicon"`, `"lexicon+llm"`), defined in `dkmem/config.py`. `dkmem_mode` must agree with the strategy (`dk-mem-lexicon` → `lexicon`, `dk-mem-lexicon-llm` → `lexicon+llm`, anything else → `off`).

### Contract changes made in the scope alignment (need Team B sign-off)
- `strategy` enum (both schemas, also `dkmem.config.STRATEGIES`): `lightmem`, `trag`, `crossrag` removed; `embedding-threshold`, `raise-tau`, `prompt-informed-judge` added; `mem0` kept (config A vs B is told apart by `pipeline_config`). `run_manifest.schema.json` now actually enforces the enum (it previously only said "same enum as pairwise_eval").
- `distinction.class` enum narrowed from 7 to 3 classes.
- `run_manifest.schema.json`: optional `pipeline_config` and `dkmem_mode`.
- `compatibility: null` now documented as "DK-Mem off".
- `tier1_eval_contract.md`, `experiment_output_spec.md` and the `teamA_*` handoff files still describe the old strategy and class lists; they were not edited.

---

## 14. Actual runs on disk — `tier1_results/`

Six completed, real runs exist (dated 2026-09-30); the two `dk-mem-lexicon-llm` strategies used Qwen2.5-3B-Instruct, the `dk-mem-lexicon` ones no model. **They predate the scope alignment** (prompt v1, the old seven-class scope, the 16-entry lexicon incl. Turkish evidentiality) and were not regenerated:

| Run directory | Strategy | Seed | Backbone | Rows in `pairwise_eval.jsonl` |
|---|---|---|---|---|
| `dk-mem-lexicon_seed{0,1,2}` | `dk-mem-lexicon` | 0,1,2 | `null` | 173 / 173 |
| `dk-mem-lexicon-llm_seed{0,1,2}` | `dk-mem-lexicon-llm` | 0,1,2 | `Qwen/Qwen2.5-3B-Instruct` | 138 / 173 |

The lexicon-only strategy produced a comparison row for every one of the 173 episodes. The LLM-assisted strategy produced rows for only 138 — the other 35 episodes failed `ExtractionError`/`ValueError` on at least one side and were correctly emitted as nothing (per §13), not as fabricated or crashed records.

No FCR/MCR/κ scoring exists in this repository — these six runs are the raw deliverable Team B scores externally (see §16).

**Consequences of the scope change for these files** (checked, nothing in `tier1_results/` was modified):
- Their manifests still validate. Of 933 pairwise rows, **135 contain a now-out-of-scope `distinction.class`** (evidentiality, classifier, politeness, deixis) and no longer validate against the narrowed `pairwise_eval.schema.json`.
- Re-running `dk-mem-lexicon` in memory on the same 173 records with the current code changes exactly **10 rows** — the 10 Turkish records, whose evidential suffix is no longer extracted (7 go `underdetermined_link` → `merge`, 3 → `no_merge`). Every other row, and every `similarity_score`, is byte-identical.
- The current Tier 1 v2 file is not re-scoped: per `tier1_eval_contract.md` only kinship (32 pairs), honorific_register (13) and name_variant (24) pairs — 69 of 173 — are in the current scope, kinship has no same-entity controls, and its labels are LLM-annotated. A new probe set is Team B's work.

---

## 15. Worked example: *chachi* vs. *mausi* (real data, not hypothetical)

Record `ep_27b02ab5a187` in the real Tier 1 input:
```
utterance_a: "meri chachi Pune mein rehti hai"   (chachi = father's younger brother's wife)
utterance_b: "meri mausi Pune mein rehti hai"    (mausi  = mother's sister)
```
Both are real lexicon entries (§7): different people, same English word ("aunt").

### `dk-mem-lexicon` (no LLM) — actual output, seed 0
```json
entry_a.gloss = "meri chachi Pune mein rehti hai"   (= raw surface, no translation)
entry_b.gloss = "meri mausi Pune mein rehti hai"
entry_a.distinction = {"kinship": "chachi"}
entry_b.distinction = {"kinship": "mausi"}
similarity_score = 0.8852459016393442
threshold        = 0.85
compatibility    = "incompatible"
decision         = "no_merge"
```
The two surface strings differ by only one word out of five, so the character-level ratio is high (0.885) and — note — **above** τ (0.85). Without gating, this pair would have been merged. The gate catches it: `kinship` is marked on both sides with different values (`chachi` ≠ `mausi`) → `incompatible` → forced `no_merge`, regardless of the (already-high) similarity.

### `dk-mem-lexicon-llm` (Qwen2.5-3B-Instruct, prompt v1) — actual output, seed 0
```json
entry_a.gloss = "user's grandmother lives in Pune"
entry_b.gloss = "user's grandmother lives in Pune"
entry_a.distinction = {"kinship": "chachi"}   (lexicon override)
entry_b.distinction = {"kinship": "mausi"}    (lexicon override)
similarity_score = 1.0
threshold        = 0.85
compatibility    = "incompatible"
decision         = "no_merge"
```
Here Qwen's own translation is actually wrong (*chachi*/*mausi* are "aunt", not "grandmother" — a real extraction error, not a cherry-picked example), and both utterances were translated to the *identical* English gloss, giving `similarity_score = 1.0` — the exact silent-false-consolidation setup the research idea describes. The lexicon overlay still correctly tagged `kinship: chachi` / `kinship: mausi` from the raw surface text (the lexicon scans `extraction.surface`, which is unaffected by the translation error), so the gate still catches the conflict and blocks the merge. This is a real, on-disk illustration of exactly why the gate is checked independently of whatever the gloss says.

---

## 16. Implemented vs. not implemented

| Component (from `DKMEM_NEW_RESEARCH_IDEA.md`) | Status |
|---|---|
| Dual-field extraction (`gloss` + `distinction` + `surface` + `lang_profile`) | **Implemented** (§6, §9) |
| Lexicon-first, model-second cascade | **Implemented** (§9) |
| Lexicon-only ablation (zero LLM calls) | **Implemented** (§13, `dk-mem-lexicon`) |
| Merge gating: `compatible()` with incompatible/underdetermined/compatible | **Implemented** (§12) |
| Gate as a veto in front of a host merge mechanism (`apply_gate`) | **Implemented** (§12); no host mechanism exists yet to put it in front of |
| Pluggable similarity separated from the gate | **Implemented** (§11) |
| Vocabulary for configurations A–D and DK-Mem off / lexicon / lexicon+llm | **Implemented as names only** (`dkmem/config.py`, manifest fields) — none of A–D runs |
| `supersede` (recognizing an update, not a duplicate) | **Not implemented** — schema value exists; `apply_gate` can veto a host's `supersede`, nothing here produces one |
| Read-side distinction-aware disambiguation (query selects among entries) | **Not implemented** — no retrieval/query code exists anywhere in the repo |
| A persistent, queryable memory store across utterances/episodes | **Not implemented** — only a verified extraction *cache* exists (§10), and Tier 1 explicitly resets per episode (§13) |
| FCR / MCR / FCR–MCR frontier computation | **Not implemented here** — Team B computes this externally from `pairwise_eval.jsonl` |
| Real embedding similarity (bge-m3) | **Not implemented** — `DIFFLIB_RATIO` is a documented stand-in; the metric interface is ready |
| Configurations A–D (real Mem0/A-Mem harness, LLM merge judge, Mem0 default native-language prompt, extractor output-language logging, verbatim storage, embedding-threshold merge) | **Not implemented** |
| Baselines: Mem0 (two prompts), A-Mem, store-surface-only, embedding-threshold, raise-τ, prompt-informed judge, flat dense RAG | **Not implemented** — only `dk-mem-lexicon` and `dk-mem-lexicon-llm` exist in code |
| Kinbank-generated kinship lexicon, name-variant entries | **Not implemented** — the lexicon is 15 Hindi entries |
| Tier 2 (injected multi-session conversations) | **Not implemented** |
| Tier 3 (one non-Indic kinship slice) | **Not implemented** — no non-Hindi lexicon entries; the `dataset_tier` enum value exists but no Tier 3 input/runner |
| Memory bloat measurement | **Not implemented** |
| Backbone sweep (1.5B/3B/7B) | **Not implemented** — only Qwen2.5-3B-Instruct has been run |
| Human-adjudicated gold labels for Tier 1 | **Not applicable yet** — current labels are explicitly "LLM-annotated, provisional" per the handoff docs |

---

## 17. Known limitations / gaps worth tracking

1. **The lexicon is small and Hindi-only.** The gate checks `kinship` and `register`, and the lexicon (§7) has 12 Hindi kinship terms and 3 Hindi register pronouns. Nothing covers other languages (German *du*/*Sie*, a non-Indic kinship slice for Tier 3) and `name_variant` has no entries; in the no-LLM strategy those can never be marked. Coverage has not been measured against any independent source.
2. **The default similarity is a character-level string metric, not a semantic one.** `DIFFLIB_RATIO` is explicitly a placeholder for a real embedding model (§11) and will behave oddly on paraphrases that are semantically identical but lexically different, or on strings that happen to share many characters without sharing meaning. `DEFAULT_TAU = 0.85` is tied to it.
3. **Only one entity per side is ever extracted.** `candidate_pairs`/`extract_lexicon_only`/`dkmem_extract_many` all assume exactly one `Extraction` per utterance side; multi-fact utterances aren't split into multiple entries anywhere.
4. **The LLM extractor can mistranslate kinship terms outright** (§15's real example: *chachi*/*mausi* → "grandmother" instead of "aunt"), which is exactly the scenario that makes the gate's independence from the gloss valuable — but it also means `gloss`-based retrieval (not implemented, but implied by the design) would be unreliable whenever this happens.
5. **35 of 173 episodes (≈20%) produce zero comparable entries in the LLM strategy** (§14) — either side's extraction failed validation. This is handled correctly (emit nothing, don't crash), but it also means roughly a fifth of the benchmark currently contributes no signal for the LLM-assisted strategy.
6. **No code enforces that `dk-mem-lexicon` and `dk-mem-lexicon-llm` use identical non-backbone settings**, as the external run-instructions contract requires (`teamA_tier1_run_instructions.md` §5) — this is a reporting/process obligation on whoever runs the experiments, not something `dkmem/tier1/runner.py` checks in code.
7. **`DISTINCTION_CLASS_TRANSLATION`'s one-key-only serialization** (§13) means an utterance that marks two distinction features simultaneously only reports one externally, even though the gate internally used both — a known, documented asymmetry between what gates the decision and what's visible in `pairwise_eval.jsonl`.
8. **Prompt v4 is untested.** It is v3 re-scoped, but it has not been run on Qwen; the 7/20 → 10/20 development-set result belongs to v2/v3. Known remaining model errors on that set (e.g. *chachi* glossed as "grandmother", wrong name script, *aunty* tagged as kinship) are not addressed by the prompt change.
9. **`gate()` and `apply_gate()` differ for underdetermined pairs below τ** (§12), and the Tier 1 runner uses `gate()`. Which semantics the A–D experiments report should be settled before bloat numbers are compared.
10. **Name variants have no merge semantics.** They are extracted and reported but neither gate nor lexicon treats them; the research plan wants same-entity name variants to merge, which needs a decision (e.g. a canonical-name key) before the secondary class can be evaluated.
11. **The Tier 1 v2 data and the six result runs are legacy-scope** (§14): 104 of 173 pairs target classes that are now out of scope, and 135 of the 933 stored rows do not validate against the current schema.

---

## 18. File-to-concept map

| File | Concept(s) it owns |
|---|---|
| `dkmem/memory/schema.py` | `ProbeItem`, `Extraction`, `MergeEvent`, JSONL I/O |
| `dkmem/memory/scope.py` | The in-scope / cannot-link / out-of-scope / legacy distinction-class sets (single source of truth) |
| `dkmem/config.py` | Names for configurations A–D, DK-Mem modes, strategy list (no behaviour) |
| `dkmem/memory/prompts.py` | Frozen extraction prompt text (v1/v2/v3 legacy, v4 final + `DEFAULT_EXTRACTION_PROMPT`) |
| `dkmem/backends/llm.py` | `HFGenerator`, `ModelConfig`, `GenerationParams` (Qwen backend) |
| `dkmem/memory/extract.py` | Baseline extraction, output validation, `derive_lang_profile` |
| `dkmem/memory/lexicon.py` | `Lexicon`, `LexiconEntry`, `load_lexicon` |
| `dkmem/memory/distinction_features.json` | The actual 15-entry lexicon data |
| `dkmem/memory/matcher.py` | Tokenizing + scanning text against the lexicon |
| `dkmem/memory/dkmem_extract.py` | Lexicon-first/model-second merge (the real "DK-Mem extractor") |
| `dkmem/memory/cache.py` | `ExtractionCache` (verified, append-only, keyed by config) |
| `dkmem/memory/similarity.py` | `gloss_similarity`, `SimilarityMetric`, `entry_similarity`, `candidate_pairs` (no gate import) |
| `dkmem/memory/consolidation.py` | `evaluate_pair` — the only place similarity and the gate are connected |
| `dkmem/memory/gate.py` | `compatible`, `gate`, `apply_gate`/`GateResult` (veto layer), `build_merge_event` |
| `dkmem/tier1/extraction.py` | `extract_lexicon_only` (no-LLM strategy) |
| `dkmem/tier1/io.py` | Input loading/validation, output schema translation, manifest/eval writers |
| `dkmem/tier1/runner.py` | `run_episode_lexicon_only` / `run_episode_lexicon_llm`, wiring everything together per episode |
| `tier1_results/*/` | The six actual completed runs (legacy scope, not regenerated) |
