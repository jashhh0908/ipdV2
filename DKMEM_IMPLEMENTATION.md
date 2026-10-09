# DK-Mem: Canonical Implementation Reference

This document describes exactly what is implemented in this repository today, file by file and function by function. It is grounded entirely in the code under `dkmem/` and the real experiment outputs under `tier1_results/` — not in the research plan. Anywhere the research plan describes something that does **not** exist in code yet, this document says so explicitly under "Not implemented".

> **Revision (scope alignment, 2026-10-07).** The research direction changed to `DKMEM_NEW_RESEARCH_IDEA.md` (stage attribution over configurations A–D). The existing code was aligned to it *without* building the A–D system: distinction scope narrowed to kinship / register / name_variant (`dkmem/memory/scope.py`); a final extraction prompt `mem0_extraction_v4` added (v1–v3 kept) and made the default; `apply_gate()` added so the gate can veto a host system's proposed merge; similarity made pluggable and separated from the gate (`dkmem/memory/consolidation.py` now holds the glue); A–D / DK-Mem ON-OFF vocabulary added (`dkmem/config.py`); the Tier 1 contracts (JSON schemas, class mapping, manifest) updated. The six runs in `tier1_results/` predate this change and were produced under the old scope (see §14).
>
> **Revision (A–D harness, 2026-10-07; Config B frozen 2026-10-08).** The stage-attribution pipeline for Configs A–D (`dkmem/pipeline/`, §13b) has the LLM merge judge (`dkmem/memory/judge.py`), the native-language extractor of Config B (`dkmem/memory/native_extract.py`, frozen prompt `native_b_v1`) and the bge-m3 embedding metric (`dkmem/backends/embedding.py`). Tested with scripted fake models, a 20-record A–C smoke test, and a 173-record × 1.5B/3B/7B check of the Config B prompt; **no full A–D experiment has been run on a real model yet**, and the baselines are not built. The first real run is specified in `DKMEM_FIRST_RUN.md`.

>
> **Revision (baselines and offline evaluation, 2026-10-08).** `dkmem/baselines/` (a hash-pinned Mem0 OSS v1.0.11 reproduction and the never-merge / flat-RAG baseline), the prompt-informed merge judge `merge_judge_informed_v1` (`dkmem/memory/judge.py`) and `dkmem/eval/` (FCR/MCR per the Tier 1 contract, the exact raise-τ sweep, the DK-Mem ablation report, a fairness check) were added; see §13c and `DKMEM_BASELINES.md`. The frozen A–D behaviour is unchanged (an optional `judge_prompt` argument defaults to `merge_judge_v1` and leaves its fingerprint unchanged). Tested with fake models only: **no baseline has been run on a real model.** The memory store (`dkmem/store/`) predates this revision and is not described in this document yet.

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
   - Lexicon found **multiple conflicting** candidate values (`_ambiguous_classes`) → **set the lexicon result aside, use the model's value for that class** (none if the model has none).
   - Lexicon found **nothing** → the class is absent. The model is **not** consulted (spec Sec 5.3: the LLM is called "only for spans the lexicon flags as ambiguous"). *Changed 2026-10-08: before, the model's value was kept for unmatched classes too; runs from that cascade are not comparable.*
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

## 13b. Stage-attribution harness (Configs A–D) — `dkmem/pipeline/`

Implements the four configurations of `DKMEM_NEW_RESEARCH_IDEA.md` §6.2, each runnable with the DK-Mem gate off, `lexicon` or `lexicon+llm`.

| Config | Extraction → stored text | Host merge decision | Isolates |
|---|---|---|---|
| **A** | `mem0_extraction_v4` (English-forced) → `gloss` | LLM judge on the glosses | L1 |
| **B** | `native_b_v1` (our own prompt: keep the user's language and script) → the one extracted fact | LLM judge on the fact | L2 |
| **C** | none → the utterance (verbatim) | LLM judge on the utterances | L3 |
| **D** | none → the utterance (verbatim) | bge-m3 cosine `> tau` | L4 |

**Per candidate pair** (one entry from utterance a × one from utterance b; exactly one pair per episode in every config) the harness records *input → extraction → stored text → similarity → host decision → gate → final decision* as one row of `trace.jsonl` (`dkmem_stage_trace_v1`, layout in the `dkmem/pipeline/runner.py` docstring), and one `pairwise_eval.jsonl` row per DK-Mem mode in the existing Team B schema.

- **Merge judge** (`judge.py`, prompt `merge_judge_v1`): sees only the stored text of each entry and answers `{"decision": "merge"|"keep_both", "reason": str}` strictly. It is deliberately neutral — no hint about kinship, register or names (the prompt-informed judge is a separate, unbuilt baseline). Because it never sees DK-Mem metadata, its decision is the same with the gate on or off, so it is made **once per pair and shared by all modes**; the gate is applied afterwards with `apply_gate` (incompatible → `keep_both`, underdetermined → `link_unresolved`, only when the host proposed a merge). Gate-off vs gate-on rows are therefore paired on identical extractions and judgements.
- **Config B is native-language extraction, not the Mem0 baseline.** `NATIVE_B_V1` (`native_extract.py`) is our own prompt, frozen on 2026-10-08 after the pre-freeze checks (`validation/native_b_prefreeze/REPORT.md`): one fact per utterance (`{"fact": str}`, strict parse, an empty or malformed fact is an extraction failure), Mem0's language sentence verbatim plus "do not translate" and "keep names and relationship terms exactly as written". Its system text is byte-identical to the validated candidate `native_b_candidate_v1` (the system-text sha256 is pinned in the tests). What B measures is what the extractor actually wrote: **language drift** (the fact is not in the source language), **script drift** (same language, another script, e.g. Hinglish → Devanagari) and **corruption** (garbled, invented or changed content), kept as separate quantities. The legacy Mem0 prompt (`MEM0_DEFAULT_FACTS_V1`, Mem0 0.1.x `FACT_RETRIEVAL_PROMPT`, verified identical to upstream) stays in the module for reference only; a real, version-pinned Mem0 run is a separate baseline that does not exist yet (Mem0 ≥ 2.0 has no LLM merge step).
- **Language drift vs script drift vs corruption.** `output_language_report` (classifier v2) labels each stored fact `source_language` / `translated_to_english` (= language drift, any language other than the source) / `script_changed` (= script drift) / `mixed` / `undetermined`; it compares *content words* only, so a surviving name or loanword is no longer evidence of the source language (validated: 93.4% agreement with hand labels on non-copy outputs, v1 51.2%). `extraction_checks` adds automatic *corruption signals* (`char_garble`: U+FFFD or a word mixing two scripts; `label_leak`) plus copy flags (`exact_copy`, `normalized_copy`) and `multi_sentence`. The signals are high precision and low recall (on the pre-freeze outputs: 20 of 20 flagged garbles were real, 32% of the hand-labelled garbles were caught, 21 flagged sides vs 400 hand-labelled corrupt ones), so **the corruption rate itself requires hand labels**; `summary.json` reports `stored_language` and `extraction_checks` side by side, never merged.
- **Config D** (`backends/embedding.py`): `BgeM3Embedder` (CLS pooling, L2-normalised) and `cosine_metric`, a `SimilarityMetric`. The score is the raw cosine; a negative cosine raises `SimilarityError` rather than being clipped. D requires an explicit metric and `tau` — there is no default threshold, because the right bge-m3 cutoff is an experimental result; the CLI accepts several (`--tau-d 0.80 0.85 0.90`), one group per value sharing the embeddings. A merge is proposed when `sim > tau`.
- **No threshold for the judge configs.** A, B and C have no similarity cutoff, so their `pairwise_eval` rows carry `threshold: null` (schema: `["number", "null"]`, needs Team B sign-off), `run_config.json` has `tau: null` / `tau_decides_merges: false`, passing a `tau` to them raises `ValueError`, and the similarity score (difflib placeholder unless a metric is passed) is informational only (`similarity.decisive: false` in the trace).
- **DK-Mem modes**: `off` (no distinctions, `compatibility: null`), `lexicon` (lexicon on the utterance), `lexicon+llm` (lexicon first; the V4 model's distinction only for a class the lexicon flags as ambiguous on that utterance, `resolve_distinction`; `lexicon_llm_policy` is recorded in the run config). In A the model's answer is the same V4 call that produced the gloss; in B/C/D a separate V4 call is made only for an ambiguous utterance (usually none; the 15-entry lexicon rarely produces two different values for one class) and is used only for the gate's distinctions (never stored, never shown to the judge). An ambiguous utterance whose V4 call fails has no distinction in this mode and its pair is skipped there.
- **Failures are never decisions.** A side whose extraction cannot be parsed (including an empty fact), a judge answer that is not valid JSON, and an unsupported language tag each produce a trace row with a `status` (`extraction_failed`, `judge_failed`, `unsupported_language`) and no `pairwise_eval` row in any mode, with the raw model output kept in the trace. An extraction failure is also a loss point: the row's `diagnosis.first_loss_point` is `dropped_at_extraction` (with `loss_stage` L1 for A, L2 for B, `sides_dropped`, and `lexicon_detectable_difference`).
- **Loss-point diagnosis** (`trace.py::diagnose`, in every trace row with a stored pair): runs the lexicon on the utterances and on each stored text. `first_loss_point` is `no_detectable_difference`, `storage` (the differing distinction is gone from a stored text **and the two stored texts have collapsed into the same text**: L1 in A, L2 in B), `host_merge` (the distinction survived storage, or the texts at least still differ, yet the host proposed a merge: L3 judge, L4 threshold) or `none`. `storage_state` is `visible`, `collapsed` or `distinct_not_visible` (the term was degraded but the texts still differ, e.g. *mami* glossed "mother" and *mausi* glossed "aunt"; reported as degraded, not as a storage loss). Lexicon-relative (gold labels are withheld from Team A); an attribution aid, not an FCR measurement.
- **Reproducibility**: `run_config.json` records the model (backbone, `run_info` with resolved revision, chat-template hash, versions), seed and all generation parameters, each prompt's id and sha256, the lexicon sha256, the similarity metric and embedding-model info, `tau`, the pipeline config, the input file sha256 and a hash over all per-pair input hashes (`pair_input_hash` excludes the gold entity ids), the git commit and dirty flag, and `code`: the sha256 of every `dkmem` source, schema and lexicon file (line endings normalized) with a tree hash, so a result can be tied to exact code on Kaggle where git is absent. `fingerprint` is the sha256 of everything that determines the outputs (not timestamps, git state or code hashes) and names the run group (`<config>-<backbone>-s<seed>-<fingerprint[:8]>`); with greedy decoding and the same inputs the trace is identical across re-runs. The CLI also writes `invocation.json` (the exact arguments).
- **Output layout**: `<out>/<group_id>/{run_config.json, trace.jsonl, summary.json, invocation.json}` and `<out>/<group_id>/<off|lexicon|lexicon-llm>/{run_manifest.json, pairwise_eval.jsonl}`. Strategy labels: gate on → `dk-mem-lexicon` / `dk-mem-lexicon-llm`; gate off → `mem0` (A, B), `store-surface-only` (C), `embedding-threshold` (D). `pipeline_config` in the manifest tells A from B. These labels name the host *role* in §6.3 of the plan; they do not mean the real Mem0 package is used (it is not), and for B in particular the prompt is not Mem0's.
- **Entry point**: `python -m dkmem.pipeline.cli` (see `DKMEM_FIRST_RUN.md` for the exact first-run command). The sanctioned Tier 1 file is hash-checked; another `--input` (e.g. the week-1 pilot pairs) is accepted and recorded as `sanctioned: false`. `--shard` splits an LLM that does not fit one T4 (Qwen2.5-7B) over both GPUs (`HFGenerator.load_sharded`). `validation/abcd_first_run/build_kernel.py` builds the Kaggle notebook that runs the unit-test suite and then this command.

Not part of this harness: the real Mem0/A-Mem packages, A-Mem, and multi-seed aggregation. The other baselines and FCR/MCR are in §13c. For Config D the τ sweep is done offline from `similarity_score` (the host decision is `sim > tau`, and the gate's effect depends only on the distinctions), as `pairwise_eval.schema.json` already assumes; the judge configs have no τ to sweep.

---

## 13c. Baselines and offline evaluation — `dkmem/baselines/`, `dkmem/eval/`

Details, exact commands and the unresolved choices are in `DKMEM_BASELINES.md`; this section is the map.

- **Prompt-informed merge judge** (`merge_judge_informed_v1`, sha256 `614f2331…`). `merge_judge_v1` (sha256 `3855ef65…`, unchanged) plus one paragraph that explains cross-lingual distinctions and tells the judge to weigh them as evidence. It is advice, not a rule: there is no "always keep both" wording, so it is not a hard cannot-link constraint (that is the DK-Mem gate). The system text before the paragraph, the two chat examples and the output format are byte-identical to v1. It runs through the A–D harness (`--judge-prompt merge_judge_informed_v1 --modes off`); the gate-off rows get the strategy `prompt-informed-judge`. The prompt id and hash enter `run_config.json` and the fingerprint.
- **Mem0 OSS v1.0.11** (`dkmem/baselines/mem0/`). Tag `v1.0.11` = commit `144627c4…`. The package is not imported: three upstream files (`memory/main.py`, `configs/prompts.py`, `memory/utils.py`) are vendored as `.py.txt`, pinned by sha256 and git blob id (the blob ids equal what GitHub reports for the tag), and verified before use; the prompts and helper functions are executed from them (with the prompt's `datetime.now()` date frozen to 2026-10-07). `Mem0Memory.add` re-implements `Memory._add_to_vector_store` for `infer=True`: one extraction call (`USER_MEMORY_EXTRACTION_PROMPT`, "record the facts in the same language"), per-fact search (`limit=5`, no threshold), id remapping to `"0".."n"`, one update call (`DEFAULT_UPDATE_MEMORY_PROMPT`), sequential ADD/UPDATE/DELETE/NONE with per-action error guards, history rows, and upstream's silent failure behaviour. Substituted: Qwen as the LLM (`response_format` cannot be enforced), bge-m3 as the embedder, in-memory vector store and history. A Tier 1 episode is two `add` calls on a fresh memory; `events.classify_pair` maps b's events to `supersede` (UPDATE/DELETE of a's memory), `keep_both` (a new memory stored), `merge` (NONE for a's memory and nothing stored), `no_entry` (an utterance gave no facts) or `host_failed` (the update step was unusable). Output: an A–D-shaped group with the single mode `off`, strategy `mem0`, and a trace holding the raw LLM input/output, retrieval hits, id mapping, every action and the history rows.
- **Never-merge / flat dense RAG** (`dkmem/baselines/flat.py`). `NeverMergeHost` in the normal `consolidate` write path, bge-m3 retrieval and the standard export: every utterance is stored verbatim, nothing is consolidated, pairwise FCR is 0 by construction (strategy `flat-dense-rag`, `backbone: null`). For Tier 2 the same store ingests a conversation (`ingest_conversation`, host-agnostic), retrieves (`FlatRAG.retrieve`, `retrieval_report`) and answers questions (`answer_questions`, prompt `flat_qa_v1`). The Tier 2 `Turn`/`Question` records are a provisional shape: the Tier 2 data does not exist yet.
- **FCR / MCR** (`dkmem/eval/metrics.py`) exactly as `tier1_eval_contract.md` §4–6: union-find over `entry_id`, denominators are all gold pairs (not-compared and no-entry pairs count as not merged), `link_unresolved` is not merged. The gold relation comes from Team B's key (`dkmem/eval/gold.py`, passed explicitly); the opaque entity ids in the sanctioned input are all distinct and are not labels.
- **Raise-τ** (`dkmem/eval/sweep.py`). Config D only: from the logged bge-m3 cosines, every distinct score plus one point below the smallest, the host decision `score > tau`, then `apply_gate` with the logged distinctions, for each DK-Mem mode — so `off` is raise-τ and `lexicon` / `lexicon+llm` are DK-Mem on the same host. Validated against real harness runs at ten τ values in the tests. Judge groups are refused: no probability/logit thresholding is implemented. Matched-MCR comparison: `best_fcr_at_mcr`, `compare_at_matched_mcr`.
- **DK-Mem ablation** (`dkmem/eval/ablation.py`): per group, lexicon-only vs lexicon+LLM FCR/MCR, gate activity (vetoes, unresolved links), extra LLM calls (0 for lexicon-only), the effect of the model fallback, and the paired-input invariants (same similarity score, stored texts and surface across modes).
- **Fairness** (`dkmem/eval/fairness.py`): reads the `run_config.json` of any set of groups and reports differing inputs, pair order, seed, backbone/revision, decoding (batch size is a note), embedder and similarity metric, lexicon hash.
- **Commands**: `python -m dkmem.baselines.cli {mem0,flat}`, `python -m dkmem.eval.cli {score,sweep,compare,ablation}`, `python -m dkmem.pipeline.cli --judge-prompt …`. `validation/abcd_first_run/build_kernel.py` can build the Kaggle notebook for each (`--baseline`, `--judge-prompt`, `--embedding-revision`) and now embeds `.txt` files (the vendored Mem0 sources).
- `strategy` values added to what `PairwiseEvalRecord` accepts (`dkmem/tier1/io.py`): `prompt-informed-judge`, `flat-dense-rag`, `raise-tau`; they were already in the schema enums. `raise-tau` is not emitted by any runner yet: the sweep is offline and writes no `pairwise_eval` rows.

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
| Gate as a veto in front of a host merge mechanism (`apply_gate`) | **Implemented** (§12) and used in front of the judge / threshold hosts of §13b |
| Pluggable similarity separated from the gate | **Implemented** (§11) |
| Vocabulary for configurations A–D and DK-Mem off / lexicon / lexicon+llm | **Implemented** (`dkmem/config.py`, manifest fields); the harness that runs them is §13b |
| `supersede` (recognizing an update, not a duplicate) | **Not implemented** — schema value exists; `apply_gate` can veto a host's `supersede`, nothing here produces one |
| Read-side distinction-aware disambiguation (query selects among entries) | **Not implemented** — no retrieval/query code exists anywhere in the repo |
| A persistent, queryable memory store across utterances/episodes | **Not implemented** — only a verified extraction *cache* exists (§10), and Tier 1 explicitly resets per episode (§13) |
| FCR / MCR / FCR–MCR frontier computation | **Not implemented here** — Team B computes this externally from `pairwise_eval.jsonl` |
| Real embedding similarity (bge-m3) | **Implemented, never run on a GPU** (`BgeM3Embedder`, `cosine_metric`; unit-tested with fake vectors). Tier 1 runs 1–6 still use the `DIFFLIB_RATIO` stand-in |
| Configurations A–D: English-forced extraction + judge (A), native-language extraction (`native_b_v1`) + judge (B), verbatim + judge (C), verbatim + bge-m3 threshold (D), each with DK-Mem off / lexicon / lexicon+llm, with a per-pair stage trace | **Implemented** (§13b), tested with fake models, a 20-record Qwen2.5-3B smoke test of A–C (`validation/smoke_abc/`) and the 173-record × 1.5B/3B/7B check of the B prompt (`validation/native_b_prefreeze/`) — **no full real-model A–D run yet** (specified in `DKMEM_FIRST_RUN.md`). Not implemented: real Mem0/A-Mem packages |
| Baselines: Mem0 (two prompts), A-Mem, store-surface-only, embedding-threshold, raise-τ, prompt-informed judge, flat dense RAG | **Implemented, never run on a real model** (§13c): a pinned Mem0 v1.0.11 reproduction (one prompt: Mem0's own default, which is not English-forced), the prompt-informed judge, the offline raise-τ sweep and the never-merge / flat-RAG control. The gate-off runs of Configs A–D remain the host-only rows for store-surface-only (C) and embedding-threshold (D) and use this repo's own prompt and judge. **Not implemented: A-Mem (dropped), LightMem (dropped), a Mem0 variant with an English-forced extraction prompt** |
| FCR / MCR, the FCR–MCR frontier, fairness check | **Implemented offline** (`dkmem/eval/`), needing Team B's gold key; tested on synthetic data only |
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
3. **Only one entity per side is extracted by the DK-Mem extractors.** `extract_lexicon_only`/`dkmem_extract_many` give one `Extraction` per utterance side. Only Config B can produce several entries per side (one per Mem0 fact), each carrying the utterance-level distinctions.
4. **The LLM extractor can mistranslate kinship terms outright** (§15's real example: *chachi*/*mausi* → "grandmother" instead of "aunt"), which is exactly the scenario that makes the gate's independence from the gloss valuable — but it also means `gloss`-based retrieval (not implemented, but implied by the design) would be unreliable whenever this happens.
5. **35 of 173 episodes (≈20%) produce zero comparable entries in the LLM strategy** (§14) — either side's extraction failed validation. This is handled correctly (emit nothing, don't crash), but it also means roughly a fifth of the benchmark currently contributes no signal for the LLM-assisted strategy.
6. **No code enforces that `dk-mem-lexicon` and `dk-mem-lexicon-llm` use identical non-backbone settings**, as the external run-instructions contract requires (`teamA_tier1_run_instructions.md` §5) — this is a reporting/process obligation on whoever runs the experiments, not something `dkmem/tier1/runner.py` checks in code.
7. **`DISTINCTION_CLASS_TRANSLATION`'s one-key-only serialization** (§13) means an utterance that marks two distinction features simultaneously only reports one externally, even though the gate internally used both — a known, documented asymmetry between what gates the decision and what's visible in `pairwise_eval.jsonl`.
8. **Prompt v4 is untested.** It is v3 re-scoped, but it has not been run on Qwen; the 7/20 → 10/20 development-set result belongs to v2/v3. Known remaining model errors on that set (e.g. *chachi* glossed as "grandmother", wrong name script, *aunty* tagged as kinship) are not addressed by the prompt change.
9. **`gate()` and `apply_gate()` differ for underdetermined pairs below τ** (§12), and the Tier 1 runner uses `gate()`. Which semantics the A–D experiments report should be settled before bloat numbers are compared.
10. **Name variants have no merge semantics.** They are extracted and reported but neither gate nor lexicon treats them; the research plan wants same-entity name variants to merge, which needs a decision (e.g. a canonical-name key) before the secondary class can be evaluated.
11. **The Tier 1 v2 data and the six result runs are legacy-scope** (§14): 104 of 173 pairs target classes that are now out of scope, and 135 of the 933 stored rows do not validate against the current schema.
12. **Config A–D limitations.** (a) Config D, the bge-m3 loading path and the CLI have only run against fakes; judge behaviour on real outputs is known only for the 20-record A–C smoke test, and A, C and the judge have not been run on all 173 records. (b) Config B's native prompt is ours, not Mem0's; at 1.5B it is effectively English-forced (92% of facts translated), at 7B 41% of facts change script (Hinglish → Devanagari); corruption (hand-labelled) was 74% / 18% / 23% for 1.5B / 3B / 7B and the automatic corruption signals catch only a fraction of it. (c) The pair-level setting removes candidate retrieval, so L4 is isolated only as "cosine of these two texts vs tau", not top-k retrieval collisions. (d) The judge answers merge/keep_both only (no supersede). (e) A–C rows have `threshold: null`; the similarity score there is informational. (f) `diagnose` is lexicon-relative and sees nothing the lexicon does not cover; a pair counts as a storage loss only when the stored texts collapsed. (g) Gate-off runs use the strategy labels `mem0` / `store-surface-only` / `embedding-threshold` for this repo's own pipeline, not the real packages. (h) The 7B model is sharded over two GPUs, whose numerics may differ slightly from single-GPU runs.

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
| `dkmem/memory/judge.py` | `MERGE_JUDGE_V1`, strict judge parser, `judge_pairs` (the LLM host for Configs A–C) |
| `dkmem/memory/native_extract.py` | `NATIVE_B_V1` (the frozen Config B prompt), fact parser, `extraction_checks`, `output_language_report` (classifier v2); legacy `MEM0_DEFAULT_FACTS_V1` |
| `dkmem/backends/embedding.py` | `BgeM3Embedder`, `cosine_metric` (Config D) |
| `dkmem/pipeline/runner.py` | `run_stage_attribution`, `build_run_config`, `write_outputs`, `strategy_for` |
| `dkmem/pipeline/trace.py` | Hashes, git state, source-file hashes, loss-point `diagnose` / `diagnose_dropped` |
| `dkmem/pipeline/cli.py` | `python -m dkmem.pipeline.cli` |
| `dkmem/baselines/mem0/{source,memory,events,runner}.py` | Pinned Mem0 v1.0.11 sources and loader; the `Memory.add` flow; event → pair-decision mapping; the Tier 1 runner and outputs |
| `dkmem/baselines/mem0/upstream/*.py.txt` | The vendored upstream files (data, hash-pinned, never imported) |
| `dkmem/baselines/flat.py` | `NeverMergeHost`, the flat-RAG Tier 1 runner, Tier 2 ingestion / retrieval / QA |
| `dkmem/baselines/cli.py` | `python -m dkmem.baselines.cli mem0 | flat` |
| `dkmem/eval/{gold,metrics,groups}.py` | The gold key, FCR/MCR per the contract, loading run groups |
| `dkmem/eval/{sweep,ablation,fairness,cli}.py` | Raise-τ sweep and frontier, DK-Mem ablation, fairness check, the evaluation CLI |
| `tier1_results/*/` | The six actual completed runs (legacy scope, not regenerated) |
