# Experiment Output Specification (for Team A)

**Audience:** Team A (memory pipeline). **Owner:** Team B (dataset / evaluation / analysis).
**Scope:** this document defines only the *output contract* Team A's pipeline must emit so Team B can
compute FCR, MCR, the FCR–MCR frontier, downstream QA accuracy, memory bloat, and the ablations
described in the research plan. It does not specify how the memory pipeline works internally —
Team A is free to implement extraction, gating, and consolidation however it wants, as long as a run
produces the two files below.

No memory-system code is implemented as part of this spec.

## What a run must emit

Every experiment run (one `strategy` × one `dataset_tier` × one `seed`, optionally one `backbone`)
emits exactly two files:

1. **`run_manifest.json`** — one object, run-level metadata. Schema: [`schemas/run_manifest.schema.json`](schemas/run_manifest.schema.json).
2. **`pairwise_eval.jsonl`** — one JSON object per line, one line per pair of memory entries the
   consolidation step compared. Schema: [`schemas/pairwise_eval.schema.json`](schemas/pairwise_eval.schema.json).

`pairwise_eval.jsonl` is the only file Team B needs for metrics. Each record is self-contained
(both entries' extraction output is embedded) so Team B never has to join against Team A's internal
storage.

## Why one pair = one record

FCR/MCR and the frontier sweep are defined over **pairs** of entries (same-entity vs.
different-entity), not over individual stored facts. Denormalizing entry data into each pairwise
record means Team B can compute every metric in the eval plan from a single file, and can re-sweep
the merge threshold τ offline (see below) without asking Team A to rerun anything.

## Completeness contract

Team A MUST emit exactly one `pairwise_eval` record for every pair of entries its consolidation
step actually compares — no fewer (a compared pair with no record is unrecoverable) and no more
(records for pairs never compared would falsely imply a comparison happened). Under this contract,
absence of a pair from `pairwise_eval.jsonl` has one unambiguous meaning: **not-compared**. This
gives Team B a clean three-way split for any gold pair:

- **compared + merged** — a record exists with `decision` in `{merge, supersede}`.
- **compared + not-merged** — a record exists with `decision` in `{no_merge, underdetermined_link}`.
- **not-compared** — no record exists for that pair at all (e.g. the system's candidate retrieval
  never surfaced the other entry, so consolidation never scored it).

Team A does not need to enumerate or explain skipped pairs — the contract makes silence
sufficient.

## Field coverage

| Required for evaluation | Where it lives |
|---|---|
| Memory extraction | `entry_a.gloss` / `entry_a.surface`, same for `entry_b` |
| Distinction information | `entry_a.distinction` (`{class, value, script, extraction_method}`) |
| Pairwise merge decision | `decision: "merge" \| "no_merge"` |
| Supersession decision | `decision: "supersede"` + `superseded_entry_id` |
| Similarity score | `similarity_score` (raw, pre-gating, always present) |
| Threshold | `threshold` (τ the system operated at) |
| Strategy / baseline | `strategy` (this record) and `run_manifest.strategy` (this run) |
| Language | `entry_a.language` / `entry_b.language` |
| Distinction class | `entry_a.distinction.class` |
| Gold entity ID | `entry_a.gold_entity_id` / `entry_b.gold_entity_id` (Team A passes through unchanged from Team B's input) |
| Predicted entity / merge ID | `predicted_entity_id` |

## Threshold-sweep contract

`similarity_score` must always be the **raw, pre-gating** similarity between `gloss_a` and
`gloss_b`, even on pairs the system blocked or never merged. This is what lets Team B recompute
`merge(a,b) := similarity_score > τ` at any τ to draw the frontier, instead of requiring Team A to
re-run the pipeline once per τ. `threshold` records the τ the system actually used to produce
`decision` for that record — kept for provenance/reproducibility, not for the sweep itself.

`compatibility` (compatible / incompatible / underdetermined) is DK-Mem's gating output and is
**independent of τ**. Baselines with no distinction gating (Mem0, A-Mem, LightMem, flat dense RAG,
store-surface-only) report `compatibility: null`.

**Limitation:** the sweep only reaches pairs that were actually compared. A not-compared pair (see
Completeness contract above) has no `similarity_score` at any τ, because it was never scored —
this is a retrieval-recall gap, not something a threshold sweep can recover. Report it separately
from gating-driven MCR misses if the distinction matters for an ablation.

## `decision` values

- `merge` — entries treated as the same entity, no temporal ordering implied.
- `no_merge` — entries kept as distinct entities.
- `supersede` — one entry replaces the other as a more recent fact about the same entity
  (`superseded_entry_id` names which one is now stale).
- `underdetermined_link` — DK-Mem's third branch: neither merged nor discarded, entries kept
  separate and cross-referenced. Used to measure memory bloat.

## Passthrough fields — Team A must not compute these

`gold_entity_id` and `source_utterance_id` originate in Team B's probe datasets (Tier 1/2/3) and
must be copied verbatim onto the corresponding entry in the output. Team A never derives or
infers these — they exist purely so Team B can score a run without re-joining against the original
input files.

## Worked example record

Based on the paper's running example: two utterances that are distinct kin terms in Hindi
(*chachi* = father's younger brother's wife, *mausi* = mother's sister) but both normalize to
"aunt" in English. Gold label: different entities. `dk-mem-lexicon` correctly blocks the merge;
a baseline without distinction gating would merge them (not shown here — that record would look
identical except `distinction: null`, `compatibility: null`, `decision: "merge"`,
`predicted_entity_id` set to a shared ID).

```json
{
  "record_id": "run_042_pair_00187",
  "run_id": "run_042",
  "strategy": "dk-mem-lexicon",
  "threshold": 0.82,
  "similarity_score": 0.94,
  "compatibility": "incompatible",
  "decision": "no_merge",
  "superseded_entry_id": null,
  "predicted_entity_id": null,
  "entry_a": {
    "entry_id": "ent_00532",
    "source_utterance_id": "tier1_probe_0091_utt_a",
    "language": "hi",
    "surface": "meri chachi Pune mein rehti hai",
    "gloss": "user's aunt lives in Pune",
    "distinction": {
      "class": "kinship",
      "value": "chachi",
      "script": "latn",
      "extraction_method": "lexicon"
    },
    "gold_entity_id": "gold_ent_kin_0091_A"
  },
  "entry_b": {
    "entry_id": "ent_00533",
    "source_utterance_id": "tier1_probe_0091_utt_b",
    "language": "hi",
    "surface": "meri mausi Pune mein rehti hai",
    "gloss": "user's aunt lives in Pune",
    "distinction": {
      "class": "kinship",
      "value": "mausi",
      "script": "latn",
      "extraction_method": "lexicon"
    },
    "gold_entity_id": "gold_ent_kin_0091_B"
  }
}
```

Corresponding `run_manifest.json`:

```json
{
  "run_id": "run_042",
  "strategy": "dk-mem-lexicon",
  "backbone": null,
  "dataset_tier": "tier1_minimal_pairs",
  "seed": 1,
  "created_at": "2026-09-22T00:00:00Z"
}
```

## What Team B computes from this (not Team A's concern, listed for context only)

- **FCR** = merged-or-superseded pairs where `gold_entity_id_a != gold_entity_id_b`, over all
  gold-different pairs.
- **MCR** = gold-same pairs left unmerged, over all gold-same pairs. A gold-same pair counts as a
  miss whether it is `no_merge`/`underdetermined_link` (compared, blocked) **or** absent from the
  file entirely (not-compared, see Completeness contract) — both mean "left unmerged." Team B
  evaluates every gold-same pair in its own probe metadata against `pairwise_eval.jsonl`; it does
  not need Team A to enumerate misses.
- **FCR–MCR frontier**: re-derive `decision` at swept τ from `similarity_score` (+ `compatibility`
  for DK-Mem variants), recompute FCR/MCR at each τ. Only covers compared pairs (see limitation
  above); not-compared pairs contribute a constant MCR term independent of τ.
- **Memory bloat**: count of `underdetermined_link` decisions per conversation, grouped by
  `run_manifest.dataset_tier`.
- **Entity clustering metrics** (B-cubed, ARI): reconstruct final clusters by **union-find over
  `entry_id`**, unioning `entry_a.entry_id` and `entry_b.entry_id` on every record where `decision`
  is `merge` or `supersede`. Do not read a record's `predicted_entity_id` in isolation as the final
  answer — because records are written incrementally as consolidation runs, an earlier record's
  `predicted_entity_id` can be superseded by a later chained merge elsewhere in the file; only the
  transitive closure over all `merge`/`supersede` edges is authoritative. An `entry_id` that never
  appears in any `merge`/`supersede` record is its own singleton cluster.
