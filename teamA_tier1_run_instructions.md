# Team A: How to run Tier 1 v2

**Audience:** Team A (memory pipeline). **Governing documents:**
[`tier1_eval_contract.md`](tier1_eval_contract.md), [`experiment_output_spec.md`](experiment_output_spec.md),
[`schemas/pairwise_eval.schema.json`](schemas/pairwise_eval.schema.json),
[`schemas/run_manifest.schema.json`](schemas/run_manifest.schema.json). If this page and the
contract disagree, the contract wins.

> Tier 1 v2 labels are **LLM-annotated/adjudicated, NOT human gold**. Report every result as
> "Tier 1 v2, LLM-annotated, provisional".

## 1. Input: one file only

Use **only** [`data/tier1/eval_input/teamA_tier1_v2.jsonl`](../data/tier1/eval_input/teamA_tier1_v2.jsonl):
173 records, sha256 `eab5dcb23375eb2e744c72a98cb31c25da0171b2ba3297a0843574ed9a34cd07`.
Check the hash before every run. If it doesn't match, stop and ask Team B.

```json
{"eval_pair_id":"ep_00d9e3e8e1dc","language":"hi","utterance_a":"Rohan ek botal doodh laaya","utterance_b":"Rohan ek gilaas doodh laaya","utterance_a_id":"ep_00d9e3e8e1dc_utt_a","utterance_b_id":"ep_00d9e3e8e1dc_utt_b","opaque_entity_id_a":"ent_0471bcb0c717c844","opaque_entity_id_b":"ent_24109d11f35eb50b"}
```

**Do not open, read, load, or let any pipeline component or LLM prompt access these files:**

- `data/tier1/candidates.jsonl` (and any other file in `data/tier1/` besides your input)
- `data/tier1/annotations/` (all files)
- `data/tier1/tier1_eval.jsonl`, `data/tier1/tier1_eval_v2.jsonl`
- `data/tier1/eval_input/teamB_answer_key.jsonl`

The utterance text can be matched against these files to recover the gold relation, so reading
any of them invalidates the run.

## 2. Running the episodes

For each of the 173 records, independently:

1. **Reset memory completely.** No entries, caches, indexes, or learned state may carry over from
   another record. Each record is its own episode.
2. **Ingest `utterance_a`**, then **ingest `utterance_b`**, in that order and as separate
   writes.
3. **Run consolidation** exactly as the strategy normally would.

Rules:

- **No manual gold or relation input.** The only information the system receives is the record
  itself: utterance text, `language` and IDs. Do not add a relation, distinction class,
  distinction value, or any hint derived from them, whether by hand, in config, or in prompts.
  Distinctions must come from the system's own extraction.
- **The IDs are passthrough only.** Don't use `eval_pair_id`, the utterance IDs or the opaque
  entity IDs as features, for example to decide a merge. The two opaque IDs always differ, by
  design, and mean nothing.
- Record order doesn't matter, because episodes are independent. Process all 173 and skip none.

## 3. Required ID mapping

Every memory entry extracted from `utterance_x` (x = `a` or `b`) must carry these values,
copied **verbatim**, in its output `MemoryEntry`:

| Input field | → Output `entry_a` / `entry_b` field |
|---|---|
| `utterance_x_id` | `source_utterance_id` |
| `opaque_entity_id_x` | `gold_entity_id` |
| `language` | `language` |

- Every extracted entry needs its own `entry_id`, and it must be **globally unique within the
  run**: no two entries may share an ID, in the same episode or across episodes. Team B runs
  union-find over `entry_id` for the whole file, so a reused ID joins unrelated entries. Keep
  the same `entry_id` for an entry in every record it appears in. Prefix the ID with the
  utterance ID, e.g. `ep_00d9e3e8e1dc_utt_a_e0`, `ep_00d9e3e8e1dc_utt_a_e1`.
- If one utterance yields several entries, each one carries that utterance's
  `source_utterance_id` and `gold_entity_id`.
- If an utterance yields no entries, emit nothing for it. Team B scores that pair as
  "no entry" (not merged).
- Team B rejects a run if any `gold_entity_id` does not match its `source_utterance_id`.

## 4. Required output (per run)

A **run** is one `strategy` × one seed, over all 173 episodes. Each run is delivered as a
directory containing exactly two files.

### `run_manifest.json`

It must validate against `run_manifest.schema.json`, with
`"dataset_tier": "tier1_minimal_pairs"`, a unique `run_id`, the `strategy`, the `seed`, and
`backbone` / `created_at` where applicable.

### `pairwise_eval.jsonl`

It must validate against `pairwise_eval.schema.json`, with one record per comparison:

- **Output every comparison actually made, and nothing else.**
  - Emit one record for each pair of entries the consolidation step actually compared, including
    comparisons that resulted in `no_merge` or `underdetermined_link`.
  - Do **not** fabricate records for comparisons that did not happen, for example to "fill in"
    the a-vs-b pair. A missing record means "not compared", and Team B scores it that way.
  - Comparisons between two entries from the same utterance are allowed if they happened, but
    Team B ignores them. Only records linking an entry from `utterance_a_id` with an entry from
    `utterance_b_id` of the same record are scored, whichever of the two is `entry_a`. A
    same-utterance merge doesn't count as a merge of the pair.
- **Preserve the raw `similarity_score`.** It is the similarity between the two glosses
  **before** any distinction gating, thresholding, or decision logic. Report it on every record,
  including blocked and non-merged ones, and do not clip, round, or overwrite it with a
  post-gating value. `threshold` records the τ actually used.
- `decision` ∈ {`merge`, `no_merge`, `supersede`, `underdetermined_link`}.
  `predicted_entity_id` is non-null for `merge` and `supersede`, and null otherwise.
  `superseded_entry_id` is set only for `supersede`.
- `compatibility` is `null` for strategies without distinction gating.
- Every record's `run_id` equals the manifest's `run_id`, and `strategy` equals the manifest's
  `strategy`.

## 5. Strategies, baselines and seeds

- Each strategy or baseline (`mem0`, `a-mem`, `lightmem`, `flat-dense-rag`,
  `store-surface-only`, `trag`, `crossrag`, `dk-mem-lexicon`, `dk-mem-lexicon-llm`) can be run
  **separately**, as its own run with its own `run_id`.
- Every run must use the **identical input file** (same sha256 as in §1), with all 173 records
  unmodified: no filtering, re-ordering-based selection, editing, or re-translation.
- Run **≥3 seeds** per strategy.
- Strategies being compared must use the **same backbone model and model settings**:
  - Record the backbone model in `run_manifest.backbone` (`null` only for strategies with no LLM
    stage).
  - Settings include decoding parameters, quantization and the embedding model used for
    `similarity_score`.
  - If a difference can't be avoided, for example a baseline that only supports a fixed model,
    send Team B the strategy, the setting and both values with the delivery, so it can be
    documented next to the results. The run manifest has no field for settings, so don't add one
    to the run files.

## 6. Pre-delivery checklist

- [ ] Input sha256 matches §1, and none of the forbidden files in §1 were accessed.
- [ ] 173 episodes run, with memory reset before each.
- [ ] Both output files validate against their schemas.
- [ ] Every entry has verbatim `source_utterance_id` and `gold_entity_id` from its source
      utterance, and `entry_id` is globally unique across the run.
- [ ] Same backbone model and settings as the other strategies being compared, or every
      unavoidable difference sent to Team B.
- [ ] `similarity_score` is raw and pre-gating on every record.
- [ ] Records exist only for comparisons that actually happened.
- [ ] No gold, relation or class information entered the system.
