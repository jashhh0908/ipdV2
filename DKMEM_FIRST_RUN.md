# First full real-model run of Configs A–D

Status: **specified and tested with fake models; not run.** Nothing below has been executed on a real model except the pieces named in "What has and has not run on real hardware".

## 1. What is frozen for this run

| Item | Value |
|---|---|
| Input | the 173 sanctioned Tier 1 records, `dkmem/data/tier1/eval_input/teamA_tier1_v2.jsonl`, sha256 `eab5dcb23375eb2e744c72a98cb31c25da0171b2ba3297a0843574ed9a34cd07` (checked on load) |
| Config A extraction | `mem0_extraction_v4` (English-forced; sha256 `3b4e1de73ac89d5142d0e0cfba52a18e03307f87ac69b2646fb1c9abbabdf00a`) |
| Config B extraction | `native_b_v1` (our native-language prompt; sha256 `4784de78239fa2a16eb18a4f604f8077ac5e40baf9c31f8d05c15a4f038525e2`; system text sha256 `709d74689e30185022400bcc855adee3fb81fc8b305d1b2ea22ce8a4ab090601`) |
| Merge judge (A, B, C) | `merge_judge_v1` (sha256 `3855ef65f50e9a0f017710e5c0d0cb5a63d210a7547fa0e0e336d4c77a45c5ff`), `max_new_tokens` 128 |
| Config C | verbatim utterance → judge; Config D | verbatim utterance → bge-m3 cosine `> tau` |
| DK-Mem modes | `off`, `lexicon`, `lexicon+llm` (all three, for every config; the gate is applied after the shared host decision) |
| Gate | `apply_gate` (veto only) with the 15-entry lexicon `dkmem/memory/distinction_features.json` |
| Backbone | Qwen2.5-3B-Instruct, weights pinned to `aa8e72537993ba99e69dfaafa59ed015b17504d1`, fp16 |
| Decoding | greedy, seed 0, batch 8, `max_new_tokens` 256 (same as every validation run) |
| Embedding (D) | `BAAI/bge-m3`, dense CLS, L2-normalised, fp16; revision not pinned yet (the resolved commit is recorded in `run_config.json` and should be pinned after this run) |
| Config D threshold | `--tau-d 0.80 0.85 0.90` (see "To confirm") |

Configs A–C carry no threshold: their rows have `threshold: null`, `run_config.tau: null`.

## 2. Exact commands

Build the Kaggle notebook (embeds the code, schemas, lexicon, Tier 1 input and the tests):

```
python validation/abcd_first_run/build_kernel.py --user jashnikumbhe --out <scratch>/kernel_3b_s0 \
    --slug dk-mem-abcd-3b-s0 --model-id Qwen/Qwen2.5-3B-Instruct \
    --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --seed 0 --tau-d 0.80 0.85 0.90
```

Save the version and run it (Internet on, 2 x T4: both are in the metadata and the push flag; the push saves version 1 and starts it):

```
kaggle kernels push -p <scratch>/kernel_3b_s0 --accelerator NvidiaTeslaT4
kaggle kernels status jashnikumbhe/dk-mem-abcd-3b-s0
kaggle kernels output jashnikumbhe/dk-mem-abcd-3b-s0 -p results/abcd_3b_s0
```

The notebook runs the full unit-test suite first (it must pass on the Kaggle image), then exactly:

```
python -m dkmem.pipeline.cli --configs A B C D --modes off lexicon lexicon+llm \
    --model-id Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
    --seed 0 --tau-d 0.80 0.85 0.90 --out /kaggle/working/runs
```

Output: five run groups (A, B, C and D at three thresholds), each `<config>-qwen-qwen2.5-3b-instruct-s0-<fingerprint8>/` with `run_config.json`, `trace.jsonl`, `summary.json`, `invocation.json`, and `off/`, `lexicon/`, `lexicon-llm/` each holding `run_manifest.json` + `pairwise_eval.jsonl`.

Rough cost (from the 20-record smoke test, scaled to 173): about 6 minutes of generation per LLM config at 3B plus model loading; D and its three thresholds add V4 calls only for lexicon-ambiguous utterances (for `lexicon+llm`; usually none) and bge-m3 embedding of 346 texts. Expect well under an hour in total.

## 3. After the run (checks, in this order)

1. `exit code 0`, unit tests `OK` in the notebook log; five groups present; `episodes_skipped_unsupported_language` 0.
2. Per group `summary.json`: `sides_extraction_failed` (A, B) and `pairs_judge_failed` (A, B, C) should be near 0; any non-zero is read from the raw outputs in `trace.jsonl` before anything else.
3. Config B: `stored_language` (source / translated = language drift / script_changed = script drift / mixed) and `extraction_checks` side by side; the corruption rate is **not** an automatic number: hand-label the non-copy outputs as in `validation/native_b_prefreeze/` (`analysis.py`, `report_stage.py`).
4. `run_config.json`: `code.tree_sha256` and per-file hashes recorded; `input.sanctioned: true`; the resolved embedding revision (pin it for the next run).
5. Hand the `pairwise_eval.jsonl` / `run_manifest.json` files of the nine (config × mode) groups to Team B; they compute FCR / MCR from the gold labels. A–C rows have `threshold: null` (schema change, needs Team B sign-off).

## 4. Next runs (same command, one change each)

```
# 1.5B
--model-id Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
# 7B (weights split over both T4s)
--model-id Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --shard
# more seeds for the 3B run (greedy decoding makes seeds identical; seeds matter once sampling or a different batch order is used)
--seed 1   --seed 2
```

## 5. What has and has not run on real hardware

| Piece | Status |
|---|---|
| Qwen2.5-3B extraction (V4), merge judge, gate, trace, outputs (A, C; B with the legacy prompt) | ran: 20-record smoke test |
| `native_b_v1` extraction, 173 records, 1.5B / 3B / 7B (7B sharded over both T4s) | ran: pre-freeze checks (extraction only, no judge) |
| bge-m3 loading and embedding, Config D end to end | **not run** |
| `python -m dkmem.pipeline.cli` itself, `HFGenerator.load_sharded` (same recipe as the pre-freeze 7B load, moved into the repo) | **not run**; CLI tested with fakes |
| Judge on all 173 pairs, Config B + judge | **not run** |

A cheap guard before the full run (not done): the same command with `--limit 10` and `--tau-d 0.85`, which exercises every untested piece in a few minutes.

## 6. To confirm before running

1. **Config D thresholds.** bge-m3 cosine scales are unknown, so `0.80 0.85 0.90` are placeholders chosen to bracket the 0.85 used so far; every pair's score is logged, so the full frontier can be computed offline for the gate-off D rows, and a different set only needs a re-run of D (LLM configs are independent of τ).
2. **Backbone for the first run:** 3B (the model all prompts were validated on). 1.5B and 7B follow with the commands above.
3. **Strategy labels** in `pairwise_eval.jsonl`: gate-off A and B are labelled `mem0` (told apart by `pipeline_config`), C `store-surface-only`, D `embedding-threshold`. B's prompt is not Mem0's, and none of these runs uses the Mem0 package. A dedicated label would be a schema change.
4. **Team B sign-off** on the contract changes (strategy and class enums, manifest fields, `threshold` nullable).
