# Baselines and offline evaluation

Status: **implemented and unit-tested with fake models; nothing below has been run on a real model.** Every number a run
will produce comes from the commands in §4. Nothing here changes the research scope of `DKMEM_NEW_RESEARCH_IDEA.md`:
no A-Mem, no LightMem, no probability/logit thresholding for Configs A–C.

## 1. The baseline set and where each row lives

| Row of Sec 6.3 | Implementation | Strategy label | Notes |
|---|---|---|---|
| Configs A–D × {off, lexicon, lexicon+llm} | `dkmem.pipeline` (frozen), `dkmem.store` | `mem0`/`store-surface-only`/`embedding-threshold` (off), `dk-mem-lexicon`, `dk-mem-lexicon-llm` | unchanged. Gate = `apply_gate` veto after one shared host decision |
| Raise τ | `dkmem.eval.sweep` (offline, Config D only) | — (no `pairwise_eval` rows) | exact: every logged score is a grid point |
| Prompt-informed merge judge | `merge_judge_informed_v1` through the A–D CLI | `prompt-informed-judge` (gate off) | prompt sha256 `614f23318c07c411135cafdc61893961d01cf3c82c2e1d4422b0ce1b467bd1ad` |
| Mem0, version-pinned, default prompt | `dkmem.baselines.mem0` | `mem0` | OSS v1.0.11, commit `144627c4ce5bc4db6acac17cbd158065f2b27a8d` |
| Flat dense RAG / never-merge | `dkmem.baselines.flat` | `flat-dense-rag` | FCR 0 by construction; `backbone: null` |
| DK-Mem lexicon-only / lexicon+LLM | modes of the A–D groups; `dkmem.eval.ablation` | `dk-mem-lexicon`, `dk-mem-lexicon-llm` | same lexicon file and hash in both |
| Mem0, English-forced extraction | **not built** | — | Sec 6.3 lists it; v1.0.11's prompt says "record the facts in the same language", so it needs a modified prompt. Not requested here |
| A-Mem, LightMem | **dropped** | — | — |

## 2. Mem0 OSS v1.0.11

**Provenance.** Tag `v1.0.11` resolves to `144627c4ce5bc4db6acac17cbd158065f2b27a8d` (GitHub API; `pyproject.toml` at the tag says 1.0.11).
Three files are vendored byte-for-byte under `dkmem/baselines/mem0/upstream/` (suffix `.py.txt`, so they are data, never imported):

| upstream path | sha256 (LF) | git blob sha1 (= GitHub's id at the tag) |
|---|---|---|
| `mem0/memory/main.py` | `04035505fc37dce54a98e5868b6775d3f57f40c4f070e6d6463138286e98e3e1` | `1804e7831dddd1c0365dddac5dbef0c3b49f7269` |
| `mem0/configs/prompts.py` | `05be41d3d0ea7e877fbde2875d146a15624435e32ca04949bc4f6304a9bd7cad` | `b7851a5105c978f301022b796511f42de10d27ef` |
| `mem0/memory/utils.py` | `0ade2e4b68122df61003ea289bbe2c86247c5dc92e18776abcb5e9fba6dcce2f` | `61e3863e380972118d297b541e6fcb1e975f6abb` |

`source.load_pinned()` verifies both hashes before it executes `prompts.py` and `utils.py` (it never executes `main.py`); the CLI also runs
`verify_structure()`, which checks that the 22 statements of `Memory._add_to_vector_store` that `Mem0Memory.add` mirrors are present in the pinned `main.py` and
that the add path contains no score threshold. The hashes and the prompt hashes are written to `run_config.json` (`mem0.files`, `prompts`).

**Preserved.** One extraction call (`USER_MEMORY_EXTRACTION_PROMPT` + `"Input:\nuser: <utterance>\n"`, `response_format={"type": "json_object"}`);
fence/`<think>` stripping, `extract_json` fallback, `normalize_facts`; a parse failure or a missing `facts` key silently becomes *no facts*; no update call when there are no facts;
per-fact bge-m3 search with `limit=5` and no threshold; de-duplication by id (last duplicate wins); ids remapped to `"0".."n"`; one update call built by the upstream
`get_update_memory_messages` (`DEFAULT_UPDATE_MEMORY_PROMPT`); an LLM exception, empty or unparseable update output becomes `{}` (no event); the returned `memory` list applied
sequentially with a per-item guard (hallucinated id, empty text, unknown event → skipped, the rest still applied); ADD / UPDATE / DELETE / NONE semantics; a history row for every create/update/delete.
An LLM exception during *extraction* is not caught upstream and is not caught here either (the run records it as `host_failed` for that episode).

**Substituted (recorded in `run_config.json` → `mem0.substitutions`).** Qwen (the run's model) for the LLM — a local model cannot enforce `response_format`, and the update call has no system
message so the chat template adds its default; bge-m3 for the OpenAI embedder; in-memory cosine search for Qdrant; a list for SQLite history; deterministic uuid5 ids and a logical clock; the
`Today's date is …` line of the extraction prompt frozen to `2026-10-07` (upstream embeds `datetime.now()` at import; the same date `native_b_v1` uses). Mem0 is sequential, so the LLM is called one prompt at a time (the A–D runs batch 8; greedy decoding, recorded as a note by the fairness check).

**Episode and event → decision mapping — FROZEN** (`events.py`, id `mem0_pair_mapping_v1`, rules text sha256 `2e10b333f63076bf47123e85acafdafc8514c48c3a16922b3ba92f758986992c`, pinned by a test and written to `run_config.json` → `baseline.mapping_sha256`). Reviewed 2026-10-08, before any result was collected, against `DKMEM_NEW_RESEARCH_IDEA.md` Sec 1 (the motivating failure is the second fact overwriting the first: "treat the second fact as an update ... the Pune fact is then deleted"), Sec 7 (FCR counts pairs the system "merges or treats as a supersession") and `tier1_eval_contract.md` Sec 4 (merged ⇔ connected by `merge`/`supersede` records; `no_merge`, `underdetermined_link`, not compared and no entry are all *not merged*). The review found the mapping consistent and changed nothing; a change from here on needs a new mapping id. Fresh memory per Tier 1 episode; `add(utterance_a)` then `add(utterance_b)`. From b's call:

| outcome | condition | `pairwise_eval` row |
|---|---|---|
| `supersede` | an applied UPDATE or DELETE hit a memory created from a | `supersede`, `superseded_entry_id` = a's entry |
| `keep_both` | b stored a new memory (applied ADD) and changed none of a's | `no_merge` |
| `merge` | b stored nothing new and the update LLM answered NONE for a's memory | `merge` |
| `no_entry` | an utterance produced no facts (empty list, or output Mem0 could not parse) | none |
| `host_failed` | update call raised / empty / unparseable / no `memory` list, or b's facts were neither stored nor matched to a | none |

Precedence `no_entry` > `host_failed` > `supersede` > `keep_both` > `merge`. Consequences to keep in mind when reading results: a benign UPDATE ("richer description of the same entity") is a `supersede`, i.e. a unification — for a gold-`different` pair it is a false consolidation, for a gold-`same` pair it is correctly merged, exactly as for `merge`; and `no_entry` / `host_failed` pairs are *not merged* (a gold-`same` pair Mem0 fails on is an MCR miss). `similarity_score` = the bge-m3 cosine (the one comparator) between a's memory text as it was when b arrived and b's entry text; `threshold: null`.

## 3. How FCR / MCR are computed and what the gold key is

`dkmem.eval.metrics` implements `tier1_eval_contract.md` §4–6 once, for every system: clusters by union-find over `entry_id` on `merge`/`supersede` records (same-utterance, cross-episode and unknown-utterance records dropped);
FCR = merged gold-`different` pairs / all gold-`different` pairs, MCR = unmerged gold-`same` pairs / all gold-`same` pairs, where *all* includes pairs a system never compared, and `underdetermined_link`
counts as unmerged. Empty denominators are `null`; cells with n < 20 are flagged `indicative`.

**Gold.** The sanctioned input's `opaque_entity_id_*` are all distinct (they are not labels). The gold format is specified in `dkmem/tier1_eval_contract.md` Sec 2: the file `data/tier1/eval_input/teamB_answer_key.jsonl`, whose fields are `eval_pair_id`, the original `pair_id`, `relation` (`same` / `different`), `distinction_class`, `language`, `utterance_a_id`, `utterance_b_id`, `opaque_entity_id_a`, `opaque_entity_id_b`. `dkmem.eval.gold` reads `eval_pair_id` + `relation` (+ optional `distinction_class`, `language`) from such a file, matching that contract; extra fields are ignored. **The key file itself is not in this repository** (Team A is blocked from it by the blinding rule), so every evaluation command takes it explicitly with `--gold` and no FCR/MCR can be computed until Team B supplies it.

**Raise-τ.** `python -m dkmem.eval.cli sweep` takes a Config D group: its log holds, per pair, the bge-m3 cosine and, per mode, the distinctions the gate saw. The host merges iff `score > tau`; `apply_gate` is applied as in the run.
Grid = every distinct score + one point below the smallest. The tests run Config D for real at ten τ values and require the sweep from a single run to reproduce each run's FCR/MCR in all three modes. `self_check` replays the run's own τ against the logged decisions.

## 4. Commands (nothing has been run)

Unit tests, then the whole set on Qwen2.5-3B (revisions as in `DKMEM_FIRST_RUN.md`; bge-m3 revision as in `dkmem/store/cli.py`):

```
python -m unittest discover -s tests

M="--model-id Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1"
E="--embedding-revision 5617a9f61b028005a4858fdac845db406aefb181"

# A-D x {off, lexicon, lexicon+llm}  (unchanged command)
python -m dkmem.pipeline.cli --configs A B C D --modes off lexicon lexicon+llm $M $E --seed 0 --tau-d 0.80 0.85 0.90 --out runs/

# prompt-informed judge (gate off): A, B, C with merge_judge_informed_v1
python -m dkmem.pipeline.cli --configs A B C --modes off --judge-prompt merge_judge_informed_v1 $M $E --seed 0 --out runs/

# Mem0 OSS v1.0.11 and never-merge / flat RAG
python -m dkmem.baselines.cli mem0 $M $E --seed 0 --out runs/
python -m dkmem.baselines.cli flat $E --seed 0 --out runs/

# offline (needs Team B's key): FCR/MCR, raise-tau, comparison + fairness, ablation
python -m dkmem.eval.cli score    --group runs/<A> runs/<B> runs/<C> runs/<D...> runs/<mem0> runs/<flat> --gold KEY.jsonl --out score.json
python -m dkmem.eval.cli sweep    --group runs/<D...> --gold KEY.jsonl --out sweep.json
python -m dkmem.eval.cli compare  --group runs/<A> runs/<B> runs/<C> runs/<mem0> runs/<flat> --d-group runs/<D> --gold KEY.jsonl --out compare.json
python -m dkmem.eval.cli ablation --group runs/<A> runs/<B> runs/<C> runs/<D...> --gold KEY.jsonl --out ablation.json
```

`validation/abcd_first_run/build_kernel.py` builds the Kaggle notebook for any of these (`--baseline mem0|flat`, or the A–D arguments with `--judge-prompt`); the notebook runs the test suite first. A cheap guard first: add `--limit 10` to any run command.

## 5. Fairness

| Held equal across systems | How it is enforced / checked |
|---|---|
| Inputs and pair order | all runners take the same record list in order; `inputs_sha256` and the trace's pair sequence are compared by `dkmem.eval.fairness` |
| Model, revision, decoding | one `GenerationParams` (greedy, 256 new tokens, seed); model id/revision/dtype and the resolved weights revision compared from `run_config.json`; `batch_size` is reported as a note |
| Embedder and similarity comparator | bge-m3 and `cosine_metric` for D, flat RAG and the Mem0 reproduction's retrieval and `similarity_score`; A–C log difflib unless a metric is passed (informational only, no decision uses it) |
| Gate | `apply_gate` everywhere, after the host decision; a gated row never differs from its `off` row except by the gate (paired-input invariants checked in the ablation report) |
| Failure policy | an unparsed extraction, an unusable host answer and an unsupported language tag are never decisions and make no row; FCR/MCR denominators are all gold pairs |
| Lexicon | one file; its sha256 is in every gated run's config and compared |
| Provenance | prompts + sha256, source-file hashes (`code`), pinned upstream hashes, git state, input hashes, fingerprint in every `run_config.json` |

**Reliability additions (multilingual evaluation audit).** Every rate carries a Wilson 95% interval (`fcr_ci95`, `mcr_ci95`); `dkmem.eval.metrics.paired_difference` compares two systems on the same pairs with an exact McNemar test, and the ablation report adds `paired_vs_off` per gated mode (the gate's effect pair by pair; at least 6 discordant pairs, all one way, are needed for p < 0.05). `score` rejects a run whose entries' `gold_entity_id` contradicts the input (contract Sec 3), and every evaluation command checks the key's language, utterance ids and opaque ids against the input (`dkmem.eval.gold.check_alignment`). `check_comparable` now also reports as issues a `lexicon+llm` policy mismatch (a run made before 2026-10-08 counts as `legacy`), one prompt id with two sha256, and a `run_manifest.json` / `pairwise_eval.jsonl` that contradicts its `run_config.json`; as warnings an unrecorded or unpinned model/embedder revision, a dirty checkout and a missing git commit; and as notes differing code trees and a non-sanctioned input. Groups are compared by the set of episodes they cover, not by trace row order (the harness writes failed episodes first). The answer key is git-ignored and `tests/test_key_protection.py` checks that nothing embedded in a Kaggle notebook holds gold labels.

`python -m dkmem.eval.cli compare` runs the check over the groups it is given and prints every disagreement. Known, unavoidable differences: the Mem0 reproduction calls the LLM one prompt at a time; its update call has no system message; flat RAG makes no LLM call.

## 6. Unresolved before the full run

1. ~~Gold key format~~ — documented in `tier1_eval_contract.md` Sec 2 and matched by `dkmem.eval.gold` (§3). **Still blocking scoring:** Team B's `teamB_answer_key.jsonl` is not available here.
2. ~~Mem0 event mapping~~ — reviewed against the spec and frozen (§2).
3. **Mem0 generation budget.** The update call returns a JSON list that grows with the number of retrieved memories; `max_new_tokens=256` (the A–D setting, kept for fairness) could truncate it. Truncation shows up as `update_failed` / `host_failed` in the trace; read those first.
4. **Local-model substitution risk.** Qwen cannot be forced to emit JSON; the 3B model may fail Mem0's extraction/update format often, which would make Mem0 mostly `no_entry`/`host_failed`. That is a result about Mem0-with-Qwen, to be reported with the failure counts, not hidden.
5. ~~DK-Mem fallback wording~~ — resolved 2026-10-08 by applying Sec 5.3 literally (§7).
6. **Tier 2.** `Turn` / `Question` and the QA prompt `flat_qa_v1` are provisional (the Tier 2 data does not exist). Tier 2 Mem0 and Tier 2 judge/threshold systems are not wired to a runner (`ingest_conversation` accepts any host).
7. **Team B sign-off** still pending on the earlier contract changes; this work adds no schema change (the three new strategy names were already in both schema enums, but `PairwiseEvalRecord` did not accept them until now).
8. **bge-m3 on a GPU, the Mem0 reproduction and the CLIs have only run against fakes.**

## 7. DK-Mem `lexicon+llm`: what the specification says, and what changed

`DKMEM_NEW_RESEARCH_IDEA.md` Sec 5.3 ("**Model fallback.** A small LLM call is made **only** for spans the lexicon flags as ambiguous") and `research_idea_context.md` ("A small LLM call **only** for spans the lexicon flags as ambiguous") agree, and Sec 5.3/5.6 call lexicon-only "zero extra inference". The existing definition of "ambiguous" is the one `dkmem_extract._ambiguous_classes` already used: the lexicon's own hits on an utterance give more than one distinct value for a class (the lexicon has no separate "ambiguous" flag). Previously `lexicon+llm` also took the model's value for every class the lexicon did not cover, in every utterance, which the spec does not allow, so it was changed:

| | before | now (`dkmem.memory.dkmem_extract.resolve_distinction`) |
|---|---|---|
| class with one lexicon value | lexicon | lexicon |
| class with conflicting lexicon values | model's value | model's value (none if the model has none) |
| class the lexicon does not cover | **model's value** | absent — the model is not consulted |
| V4 call for the gate in Configs B/C/D | every utterance | only an utterance with an ambiguous class; a failed call leaves that utterance without a distinction and its pair skipped in that mode |
| Config A | host's V4 call used as the model | same call, but its distinction is read only for ambiguous classes |

Effects to expect: on an input with no ambiguous utterance `lexicon+llm` equals `lexicon` (and costs no extra calls); a term outside the 15-entry lexicon (e.g. *bhaiya*) is never tagged; `name_variant` (no lexicon entries) no longer appears in `lexicon+llm` rows' `distinction` (it never gated). `run_config.json` records `lexicon_llm_policy: model_only_for_lexicon_ambiguous_classes_v1` for runs with the mode, so the fingerprint differs from any earlier run, and each trace side logs `lexicon_ambiguous`. Runs made before this change (e.g. a `results/abcd_3b_s0` produced earlier) used the old cascade and are not comparable; `tests/test_store_regression.py` skips them. The legacy Tier 1 runner's `dk-mem-lexicon-llm` strategy shares `apply_lexicon` and follows the same rule.
