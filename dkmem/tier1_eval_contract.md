# Tier 1 Evaluation Contract (provisional set v2)

**Owner:** Team B. **Consumer:** Team A. **Applies to:** runs with
`run_manifest.dataset_tier == "tier1_minimal_pairs"` scored against
[`data/tier1/tier1_eval_v2.jsonl`](../data/tier1/tier1_eval_v2.jsonl)
(173 pairs, sha256 `3310956620bd4205e13fcb61fd622330a322dbad9c93a38c37de2309d5b604cb`).

> **Status: LLM-annotated/adjudicated, NOT human gold.** Every `relation` label in
> `tier1_eval_v2` comes from two LLM annotation passes plus LLM adjudication. No native speaker
> has reviewed them (see [`data/tier1/README_eval_v2.md`](../data/tier1/README_eval_v2.md)).
> Label every FCR/MCR number computed under this contract **"Tier 1 v2, LLM-annotated,
> provisional"**. Do not present it as a validated benchmark result.

This contract specializes [`experiment_output_spec.md`](experiment_output_spec.md) and
[`schemas/pairwise_eval.schema.json`](schemas/pairwise_eval.schema.json) for Tier 1. It does not
change either one. Where this document is silent, the spec governs.

## 1. Input pair fields (Team B → Team A)

**Team A input:** [`data/tier1/eval_input/teamA_tier1_v2.jsonl`](../data/tier1/eval_input/teamA_tier1_v2.jsonl),
173 records. Each record is one **independent episode**: reset memory, ingest `utterance_a`,
then `utterance_b`, then run consolidation. Each record has exactly these fields:

| Field | Notes |
|---|---|
| `eval_pair_id` | Random ID (`ep_` + 12 hex). It is **not** the original `t1_XXXX` pair_id. |
| `language` | `hi`, `tr`, `de`, `ko` |
| `utterance_a`, `utterance_b` | Ingest in this order |
| `utterance_a_id`, `utterance_b_id` | `{eval_pair_id}_utt_a`, `{eval_pair_id}_utt_b` |
| `opaque_entity_id_a`, `opaque_entity_id_b` | Random (`ent_` + 16 hex), always different for A and B, and unique across the file |

All IDs are drawn at random. None is derived from the original pair_id or the relation, and
records are in random order. Team A does **not** receive the original pair_id, `relation`,
`distinction_class`, `distinction_value_*` or `provenance`. Distinctions must come from the
system's own extraction.

**Blinding rule:** the utterance text can still be matched against Team B's files. Anyone
running Team A's pipeline must not read `candidates.jsonl`, `tier1_eval_v2.jsonl`,
`data/tier1/annotations/` or `teamB_answer_key.jsonl`.

## 2. Gold relation field

- **Gold field:** `relation` (`same` / `different`). It lives only in
  [`data/tier1/eval_input/teamB_answer_key.jsonl`](../data/tier1/eval_input/teamB_answer_key.jsonl),
  which copies it from `tier1_eval_v2.jsonl`. Nothing else is gold. `gold_relation` in
  `candidates.jsonl` is a generation-time default and is **never** used for scoring.
- **Key fields:** `eval_pair_id`, the original `pair_id` (`t1_XXXX`), `relation`,
  `distinction_class`, `language`, `utterance_a_id`, `utterance_b_id`, `opaque_entity_id_a` and
  `opaque_entity_id_b`. The key maps `eval_pair_id` ↔ `pair_id` exactly 1:1.
- **Scoring join:** output `entry.source_utterance_id` → key `utterance_a_id` /
  `utterance_b_id` → `pair_id` and `relation`. `gold_entity_id` is only a passthrough check. It
  is one opaque token per utterance, so it carries no same/different signal.
- **Regeneration:** the inputs are built by `data/tier1/scripts/build_tier1_eval_inputs.js`. A
  rebuild reuses the random mapping stored in the key. Regenerating with `--regenerate` makes any
  existing Team A outputs unscoreable.

## 3. Prediction format (Team A → Team B)

Use the standard run output: one `run_manifest.json` and one `pairwise_eval.jsonl`, both
schema-valid. Tier 1 adds these requirements:

- `run_manifest.dataset_tier = "tier1_minimal_pairs"`.
- **Required field mapping.** Every memory entry extracted from `utterance_x` (x = a or b) must
  carry these values in its output `MemoryEntry`, copied verbatim:

  | Input field (Team A file) | → Output field (`entry_a` / `entry_b`) |
  |---|---|
  | `utterance_x_id` | `source_utterance_id` |
  | `opaque_entity_id_x` | `gold_entity_id` |
  | `language` | `language` |

  The output schema has no field for `eval_pair_id`. It is recoverable from
  `source_utterance_id`.
- **`entry_id` is globally unique within a run.** Every extracted memory entry gets its own
  `entry_id`, and no two entries anywhere in the run's `pairwise_eval.jsonl` may share one, even
  across different episodes. Team B runs union-find over `entry_id` for the whole file, so a reused
  ID would join unrelated entries. An entry that appears in several records keeps the same
  `entry_id` in each (the schema requires `entry_id` to be stable within a run). Prefixing with
  `source_utterance_id` (e.g. `ep_00d9e3e8e1dc_utt_a_e0`) satisfies both rules.
- `similarity_score` is raw and pre-gating, and it is always present.
- `compatibility` is `null` for strategies without gating.
- Follow the spec's completeness contract: emit one record for every comparison made, and none
  for comparisons not made.
- Records pairing entries from two different `eval_pair_id`s should not occur, because episodes
  are isolated. If they do occur, they are ignored.
- **Same-utterance records are ignored.** A record whose `entry_a` and `entry_b` have the same
  `source_utterance_id` (two entries from one utterance, or an entry compared with itself) is not
  scored. Only links between the pair's two distinct utterance IDs (`utterance_a_id` and
  `utterance_b_id` of the same `eval_pair_id`) count, in either orientation: which one appears as
  `entry_a` doesn't matter.
- Records whose `source_utterance_id` is not in the key are ignored.
- Team B rejects a run if any entry's `gold_entity_id` does not match its `source_utterance_id`
  in the key.

## 4. Merged / not-merged interpretation

For each pair, Team B builds clusters by union-find over `entry_id`. It unions every record whose
`decision ∈ {merge, supersede}`, as the spec requires, after dropping the ignored records in §3
(same-utterance, cross-episode, unknown `source_utterance_id`). Dropping same-utterance records
never changes an outcome: any merge path from a `utt_a` entry to a `utt_b` entry must contain a
cross-utterance edge. A **cross-utterance record** is one with an entry from `utt_a` and an
entry from `utt_b` of the same pair, in either orientation. The pair is then:

| Outcome | Condition | Counts as |
|---|---|---|
| **merged** | Some entry from `utt_a` and some entry from `utt_b` end up in the same cluster | merged |
| **compared, not merged** | A cross-utterance record exists, but no cross-utterance merge path exists (`no_merge` / `underdetermined_link`) | not merged |
| **not compared** | No record links an `utt_a` entry to a `utt_b` entry | not merged |
| **no entry** | An utterance produced zero entries | not merged |

`underdetermined_link` counts as **not merged**.

## 5. FCR (false consolidation rate), gold `different`

```
FCR = #{pairs with relation == "different" that are merged} / #{pairs with relation == "different"}
```

The denominator is **all** gold-`different` pairs in `tier1_eval_v2` (98 overall), including
pairs that were not compared or produced no entry. Lower is better.

## 6. MCR (missed consolidation rate), gold `same`

```
MCR = #{pairs with relation == "same" that are not merged} / #{pairs with relation == "same"}
```

The denominator is **all** gold-`same` pairs in `tier1_eval_v2` (75 overall). Every not-merged
outcome in §4 is a miss. Lower is better.

For the FCR–MCR frontier, Team B re-sweeps τ offline from `similarity_score`, following the spec.
Not-compared and no-entry pairs are a constant MCR term at every τ.

## 7. Ambiguous and excluded pairs

- The 53 candidates excluded from v2 are **not scored** and do not appear in any FCR or MCR
  numerator or denominator. All 53 were excluded as `ambiguous`; none were excluded for
  grammaticality or minimal-pair reasons.
- Team B does not send them as Team A input.
- A merge rate on the ambiguous items may be reported as a clearly separate descriptive
  diagnostic only. It must never be folded into FCR or MCR.

## 8. Required reporting

**Comparability.** Strategies compared against each other must use the same backbone model
(recorded in `run_manifest.backbone`) and the same model settings, for example decoding
parameters, quantization and the embedding model behind `similarity_score`. A strategy with no
LLM stage (`backbone: null`, e.g. lexicon-only DK-Mem) is exempt from the backbone rule only.
If a difference can't be avoided, for example a baseline that only supports a fixed model,
document the strategy, the setting and both values in the results report, next to every table
that compares those strategies. The run manifest has no field for model settings, so this goes
in the report, not in the run files.

For each `strategy`, report the mean ± sd over ≥3 seeds per the manifest schema:

1. **Overall:** FCR and MCR, each with its denominator (n = 98 / 75).
2. **Per class:** FCR and MCR per `distinction_class`, with n.
3. **Per language:** FCR and MCR per `language`, with n.
4. **Class × language:** FCR and MCR per cell, with n.
5. **Outcome breakdown:** counts of merged / compared-not-merged / not-compared / no-entry,
   split by gold relation.

Reporting rules:

- If a cell has **n = 0 for a relation, report `n/a`**, not 0. In v2 this applies to:
  - **FCR:** `evidentiality`; `tr`.
  - **MCR:** `kinship`, `classifier_measure`, `spatial_temporal_deixis`; `de`.
- **Flag any cell with n < 20 as indicative.** In v2 this includes every `de`, `ko` and `tr`
  cell, and `honorific_register` (13).
- **Do not headline overall FCR/MCR alone.** Class and relation are largely confounded, and
  `honorific_register` is all `same` in Hindi and all `different` in German. 145 of the 173 pairs
  are Hindi.

Gold counts in v2, for checking denominators:

| distinction_class | same | different | | language | same | different |
|---|---|---|---|---|---|---|
| evidentiality | 34 | 0 | | hi | 57 | 88 |
| kinship | 0 | 32 | | ko | 8 | 4 |
| name_variant | 20 | 4 | | tr | 10 | 0 |
| classifier_measure | 0 | 24 | | de | 0 | 6 |
| politeness_relationship | 14 | 10 | | | | |
| spatial_temporal_deixis | 0 | 22 | | | | |
| honorific_register | 7 | 6 | | | | |
| **total** | **75** | **98** | | | | |
