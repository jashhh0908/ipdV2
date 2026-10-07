# Configs A-C real-model smoke test (20 Tier 1 records)

**Run:** Kaggle notebook `jashnikumbhe/dk-mem-abc-smoke-test`, version 1 (pushed = saved, then run), private, Internet on, accelerator "GPU T4 x2" (`nvidia-smi -L` listed 2 x Tesla T4; decoding on GPU 0 only, as in earlier runs). Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, fp16, greedy, seed 0, batch 8, max_new_tokens 256 (judge 128), torch 2.11.0+cu128, transformers 5.16.1. Frozen V4 prompt (sha unchanged, checked by the embedded unit tests, which passed on Kaggle). Input: first 20 lines of `teamA_tier1_v2.jsonl` (slice sha256 `fb2fd613...`, parent file sha256 `eab5dcb2...`, recorded as `sanctioned: false` because the full-file hash check cannot run on a slice). Each config ran in all three DK-Mem modes. No Config D, no bge-m3, no baselines.

**Mem0 prompt check (before B):** `mem0_default_facts_v1` was compared programmatically with `FACT_RETRIEVAL_PROMPT` in `mem0/configs/prompts.py` at tag v1.0.0 and at main (braces un-escaped, date line frozen): identical. The user turn is `"Input:\n" + parse_messages(...)`, which ends in a newline; ours lacked it, so the template was corrected to `Input:\nuser: {utterance}\n` before the run (the prompt had never been run; new sha256 `e9c4726f...`, now pinned by a test). Upstream copies used are in this folder.

## Results

| Check | A | B | C |
|---|---|---|---|
| Host extraction parsed (sides) | 40/40 | 40/40 parsed; 26 returned `{"facts": []}`, 14 with a fact | n/a (verbatim) |
| V4 extraction parsed (sides; gate fallback / A's gloss) | 40/40 | 40/40 | 40/40 |
| Candidate pairs reaching the judge | 20/20 | **3/20** | 20/20 |
| Judge answers valid | 20/20 | 3/3 | 20/20 |
| Judge decisions (merge / keep_both) | 11 / 9 | 1 / 2 | 15 / 5 |
| Trace rows complete (all stages present) | 20/20 | 20/20 (17 are `no_entries`) | 20/20 |
| `pairwise_eval` / manifest valid against Team B schemas, rows = trace, run ids consistent (3 modes) | yes | yes | yes |
| `you`/`My` gloss violations | 0/40 | n/a | n/a |
| Gate vetoes (lexicon / lexicon+llm) | 0 / 0 | 0 / 0 | 0 / 1 |
| Extraction/judge errors, exceptions | 0 | 0 | 0 |

Other: unit tests passed on Kaggle; Config A re-run gave identical fingerprint, group id, trace and `pairwise_eval` rows in all modes; time 37 / 46 / 40 s per config, peak GPU memory 7.5 GB; the Kaggle log has only generic warnings (HF token, notebook-id, debugger).

## Config B: native-language extraction

- **65% of sides (26/40) produced no fact.** Mem0's prompt extracts the *user's* personal facts and preferences; most Tier 1 sentences are third-person events ("Rohan ek botal doodh laaya"), so 17 of 20 pairs have a side with nothing stored and produce no pair to judge or gate.
- **No fact kept the source language.** Of the 14 facts: 10 `translated_to_english`, 4 `mixed`, 0 `source_language` (e.g. "Rohan bought a carton of milk", "Neha was at a party"), despite "record the facts in the same language". Heuristic labels from `output_language_report`; the sample is small.
- Mem0's own fence/format handling was never needed (0 fences, 0 invalid JSON).

## Config A / C observations (not accuracy claims)

- A's English glosses lose or distort detail beyond the targeted distinctions: "gilaas doodh" -> "drank some strange milk", "achaar bheja" -> "cooked well", "botal" -> "carton", *mami* -> "user's mother" (a mistranslation, as with *chachi* earlier), and an invented "to user" in two glosses. The one lexicon-detectable pair (*mami* vs *mausi*) is marked `storage` (L1), but the glosses ("mother" vs "aunt") differ by luck and the judge kept them apart.
- C's judge merged 15/20 verbatim pairs, including clearly different events (*gaya tha* vs *jayega*, past vs future). C's one gate action came from the `lexicon+llm` fallback (V4 tagged *bhaiya* as kinship; *Verma uncle* vs *Verma bhaiya* became `underdetermined_link`).
- Only 1 of 20 pairs is a lexicon-detectable kinship conflict (the first 20 records are mostly legacy-scope classifiers, tense, evidentiality), so this slice says almost nothing about gate effects.

## Issues to settle before the full 173-record run

1. **Config B drops most of the data and does not keep the native language** (above). Decide what B is: (a) keep the Mem0 prompt as is and report empty extractions as a result, with the consequence that B measures little on third-person sentences; or (b) use a Mem0 prompt variant / wrap utterances so facts are extracted (changes what "Mem0 default" means). Related: **this prompt is the legacy Mem0 0.1.x extractor.** From Mem0 1.0 the default is `USER_MEMORY_EXTRACTION_PROMPT` plus an LLM update step; from 2.0.0 (current 2.2.1) it is a single additive extraction with no LLM merge step, and its `use_input_language` option defaults to False. The "Mem0 default keeps the user's language" statement in the plan needs checking against the version being cited.
2. **Trace has no loss-point label for dropped pairs.** `no_entries` rows have no `stored`/`diagnosis`, so "lost at extraction" is only visible as a status. Add an explicit `extraction_dropped` loss point for B.
3. **Code provenance is empty on Kaggle** (`git: {commit: null, dirty: null}`); only the notebook's per-file sha256 in `run_environment.json` identifies the code. Record the file hashes in `run_config.json` (or pass the commit in).
4. **`threshold` in A-C rows is a nominal 0.85** the judge never uses; Team B should be told not to read it as the cutoff (`run_config.json` says `tau_decides_merges: false`).
5. **`diagnose` calls a pair `storage`-lost even when the stored texts still differ** (A's *mami*/*mausi* glosses), because it tests whether the lexicon value survives, not whether the texts differ. Keep, but read it with `host_proposed`.
6. The 0.85 / difflib similarity logged for A-C is the placeholder, not an embedding.
7. Not tested here: Config D, bge-m3 loading, the CLI, more than one seed, 1.5B/7B.

## Files

`kaggle_output/runs/<group>/{run_config.json, trace.jsonl, summary.json, off|lexicon|lexicon-llm/{run_manifest.json, pairwise_eval.jsonl}}` for A, B, C; `runs_repeat/` (Config A re-run); `determinism.json`; `run_environment.json`; `tier1_slice.jsonl`; the Kaggle log; `metrics.json` (output of `check_smoke.py`); `build_kernel.py`; `mem0_*_upstream.py`, `prompts_v1.0.0.py`, `main_v1.0.0.py` (upstream Mem0 sources used for the prompt check).
