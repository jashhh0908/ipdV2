# Team A: Tier 1 v2 handoff

**Audience:** Team A. **Sources (the only ones used):**
- [`teamA_tier1_run_instructions.md`](teamA_tier1_run_instructions.md) (RI)
- [`tier1_eval_contract.md`](tier1_eval_contract.md) (C)
- [`teamB_tier1_run_intake_checklist.md`](teamB_tier1_run_intake_checklist.md) (CL)

Citations such as "RI §4" point to those docs. If anything here disagrees with them, the sources
win, and the contract wins over both of the others (RI intro, CL intro).

- **Required** = what Team A must do (RI, C).
- **Intake-checklist BLOCK** = something Team A must send or satisfy because Team B's intake
  checklist blocks the run without it. It comes from CL only; RI and C don't require it.
- **Checklist validation** = what Team B checks on delivery (CL). **BLOCK** = the run isn't scored
  or reported until it's fixed (CL intro); see the threshold-sweep note in §7. **WARN** = scored,
  with the warning recorded.

> Tier 1 v2 labels are LLM-annotated/adjudicated, not human gold. Report every result as
> "Tier 1 v2, LLM-annotated, provisional" (RI, C).

## 1. Input (Required)

- Use **only** [`data/tier1/eval_input/teamA_tier1_v2.jsonl`](../data/tier1/eval_input/teamA_tier1_v2.jsonl):
  173 records, sha256 `eab5dcb23375eb2e744c72a98cb31c25da0171b2ba3297a0843574ed9a34cd07`.
  Check the hash before every run. If it doesn't match, stop and ask Team B. (RI §1)
- Each record has exactly these fields: `eval_pair_id`, `language`, `utterance_a`, `utterance_b`,
  `utterance_a_id`, `utterance_b_id`, `opaque_entity_id_a`, `opaque_entity_id_b`. (C §1)
- **Never open, read, load, or let any pipeline component or LLM prompt access these files**
  (RI §1, C §1):
  - `data/tier1/candidates.jsonl`, or any other file in `data/tier1/` besides your input;
  - `data/tier1/annotations/`;
  - `data/tier1/tier1_eval.jsonl`, `data/tier1/tier1_eval_v2.jsonl`;
  - `data/tier1/eval_input/teamB_answer_key.jsonl`.

  Reading any of them invalidates the run.
- Use the same input file, unmodified, for every run: no filtering, selection by order, editing
  or re-translation. (RI §5)

## 2. Running the episodes (Required)

For each of the 173 records, independently (RI §2, C §1):

1. Reset memory completely. No entries, caches, indexes or learned state carry over.
2. Ingest `utterance_a`, then `utterance_b`, in that order, as separate writes.
3. Run consolidation exactly as the strategy normally would.

Rules:
- The system receives only the record: the utterance text, `language` and the IDs. No relation,
  distinction class, distinction value or hint derived from them, by hand, in config or in prompts.
- The IDs are passthrough only; never use them as features.
- Process all 173 records and skip none. (RI §2)

## 3. Outputs per run (Required)

A **run** is one `strategy` × one seed over all 173 episodes, delivered as **a directory containing
exactly two files**: `run_manifest.json` and `pairwise_eval.jsonl`. (RI §4, C §3)

### `run_manifest.json`
- Validates against `run_manifest.schema.json`. (RI §4)
- `"dataset_tier": "tier1_minimal_pairs"`, a unique `run_id`, the `strategy`, the `seed`, and
  `backbone` / `created_at` where applicable. (RI §4, C §3)
- `backbone` is the backbone model, and is `null` only for strategies with no LLM stage. Don't add
  a settings field; the manifest has none. (RI §5)

### `pairwise_eval.jsonl`
- Validates against `pairwise_eval.schema.json`, with **one record per comparison actually made
  and nothing else**, including `no_merge` and `underdetermined_link` outcomes. Never fabricate
  records. A missing record means "not compared". (RI §4, C §3)
- **ID mapping:** every entry extracted from `utterance_x` (x = `a` or `b`) carries these, copied
  verbatim (RI §3, C §3):

  | Input field | → Output `entry_a` / `entry_b` field |
  |---|---|
  | `utterance_x_id` | `source_utterance_id` |
  | `opaque_entity_id_x` | `gold_entity_id` |
  | `language` | `language` |

- **`entry_id`:**
  - globally unique within the run, across all episodes;
  - the same for an entry in every record it appears in;
  - prefixed with the utterance ID, e.g. `ep_00d9e3e8e1dc_utt_a_e0`.

  (RI §3, C §3)
- Several entries from one utterance each carry that utterance's `source_utterance_id` and
  `gold_entity_id`. An utterance with no entries emits nothing, and is scored "no entry". (RI §3)
- `decision` ∈ {`merge`, `no_merge`, `supersede`, `underdetermined_link`}.
  `predicted_entity_id` is non-null for `merge` / `supersede` and null otherwise.
  `superseded_entry_id` is set only for `supersede`. (RI §4)
- Every record's `run_id` and `strategy` equal the manifest's. (RI §4)
- **Which records are scored:**
  - Same-utterance comparisons are allowed, but ignored.
  - Only records linking an `utterance_a_id` entry with an `utterance_b_id` entry of the same
    record are scored, in either orientation.
  - Records pairing two different `eval_pair_id`s should not occur, and are ignored.

  (RI §4, C §3)

## 4. Multi-seed runs (Required)

- Run **at least 3 seeds per strategy**. Each strategy or baseline × seed is its own run with its
  own `run_id`. (RI §5)
- Strategies being compared use the **same backbone model and model settings**: decoding
  parameters, quantization, and the embedding model used for `similarity_score`. (RI §5, C §8)
- **Exemption:** a strategy with no LLM stage (`backbone: null`, e.g. lexicon-only DK-Mem) is exempt
  from the backbone rule only. Its other model settings must still match. (C §8)
- If a difference can't be avoided, send Team B the strategy, the setting and both values with the
  delivery. (RI §5)

## 5. Threshold sweep (Required inputs; Team B runs the sweep)

Team B re-sweeps τ offline from `similarity_score` (C §6). The sources don't ask Team A to run a
sweep; they ask for the fields below.

- `similarity_score` is the **raw, pre-gating** similarity between the two glosses, on every
  record, including blocked and non-merged ones. Never clip, round, or overwrite it with a
  post-gating value. (RI §4, C §3)
- `threshold` records the τ actually used. (RI §4)
- `compatibility` is `null` for strategies without distinction gating. (RI §4, C §3)

## 6. What to send back to Team B

Each item is labeled with where the obligation comes from.

1. **Required: one directory per run** (strategy × seed), each holding exactly
   `run_manifest.json` and `pairwise_eval.jsonl`, for at least 3 seeds per strategy. (RI §4–§5)
2. **Intake-checklist BLOCK: confirmation** that every run used
   `data/tier1/eval_input/teamA_tier1_v2.jsonl` with the sha256 above, 173 records, unmodified.
   (CL §2)
3. **Intake-checklist BLOCK: attestation** that no pipeline component, prompt or person accessed
   the forbidden files in §1. (CL §2)
4. **Intake-checklist BLOCK: attestation** that memory was reset before each of the 173 episodes,
   each episode ingested `utterance_a` then `utterance_b`, and no gold, relation, class or
   distinction hints were given to the system. (CL §2)

   *Items 2–4 are sent because CL §2 blocks the run without them. RI and C don't require sending
   them; RI §6 lists the same conditions only as pre-delivery self-checks. The underlying rules
   themselves are required (RI §1, §2, §5; C §1).*
5. **Required: every unavoidable backbone or model-setting difference** between compared
   strategies: the strategy, the setting and both values, sent with the delivery. (RI §5; C §8)
6. **Intake-checklist BLOCK: the model settings for each run.** These are the decoding
   parameters, quantization, and the embedding model used for `similarity_score` (RI §5, C §8).
   - **Why:** Team B needs them to check that compared strategies use the same backbone and model
     settings. The checklist says to "get the settings from Team A". (CL §4)
   - **How to send:** with the delivery, not inside the run files; the run manifest has no field
     for settings. (RI §5, C §8)
   - **Granularity:** CL §4 doesn't say whether settings are needed per run or per strategy.
     Sending them for each run covers both.

**Pre-delivery self-check (Required, RI §6):**
- [ ] The input sha256 matches, and none of the forbidden files were accessed.
- [ ] 173 episodes run, with memory reset before each.
- [ ] Both output files validate against their schemas.
- [ ] Every entry has verbatim `source_utterance_id` and `gold_entity_id`, and `entry_id` is
      globally unique across the run.
- [ ] Same backbone and settings as the other strategies being compared, or every difference sent.
- [ ] `similarity_score` is raw and pre-gating on every record.
- [ ] Records exist only for comparisons that actually happened.
- [ ] No gold, relation or class information entered the system.

## 7. Checklist validation (what Team B checks on delivery)

**BLOCK** (not scored or reported until fixed, per CL intro; for the threshold-sweep items, see the
note at the end of that group):
- **Delivery:** not exactly two files in the run directory; a `run_id` that isn't unique across
  delivered runs. (CL §1)
- **Manifest:** fails the schema. Required fields are `run_id`, `strategy`, `dataset_tier` and
  `seed` (an integer); `created_at`, if present, must be an RFC 3339 date-time. Or `dataset_tier`
  isn't `"tier1_minimal_pairs"`, or `strategy` isn't one of `mem0`, `a-mem`, `lightmem`,
  `flat-dense-rag`, `store-surface-only`, `trag`, `crossrag`, `dk-mem-lexicon`,
  `dk-mem-lexicon-llm`. (CL §1)
- **Missing confirmation or attestations:** any of §6 items 2–4. (CL §2)
- **Pairwise records:**
  - a line isn't valid JSON or fails the schema;
  - a record's `run_id` or `strategy` doesn't match the manifest;
  - `record_id` is repeated;
  - one `entry_id` is used for entries from different `source_utterance_id`s;
  - an entry's `gold_entity_id` doesn't match its `source_utterance_id`.

  (CL §3, C §3)
- **Multi-seed:**
  - fewer than 3 distinct seeds;
  - a duplicate `seed` or `run_id`;
  - `strategy`, `dataset_tier` or `backbone` differ across a strategy's runs (a missing `backbone`
    counts as `null`), or any other manifest key present in every run differs. Only `run_id`,
    `seed` and `created_at` may differ;
  - one invalid run blocks that strategy's whole summary;
  - compared strategies don't use the same backbone and model settings. Team B checks this using
    the settings from §6 item 6.

  (CL §4)

  **Precedence on settings:** CL §4 states the rule as "use the same backbone and model settings".
  C §8 allows an unavoidable difference if it's documented next to every table comparing those
  strategies, and exempts no-LLM strategies from the backbone rule only (§4 above). Where they
  differ, the contract wins (RI intro, CL intro).
- **Threshold sweep:**
  - a scored (cross-utterance) record has no `compatibility` field (it must be `null` or a value);
  - a run mixes `null` and non-null `compatibility`;
  - a `dk-mem-lexicon` / `dk-mem-lexicon-llm` run has `compatibility: null` on every scored
    record.

  (CL §5)

  **Note:** the sources (RI, C, CL) do not settle whether a failure on only these threshold-sweep
  items (CL §5) also blocks ordinary single-run or multi-seed scoring.

**WARN** (scored, with the warning recorded):
- An entry's `language` differs from the input `language`. (CL §3)
- A `supersede` record's `superseded_entry_id` isn't `entry_a` or `entry_b`. (CL §3)
- A manifest key present in only some of a strategy's runs; it isn't compared. (CL §4)
- At a record's own `threshold`, the sweep rule disagrees with the system's `decision`. This is
  expected for LLM-judged strategies. (CL §5)

**Ignored, with counts reported** (CL §3, C §3):
- records with an unknown `source_utterance_id`;
- cross-episode records;
- same-utterance records.

A high count prompts Team B to ask Team A about a mapping or ID bug. Team B also checks
completeness manually: a very large `no_entry` count can indicate skipped episodes. (CL §3)
