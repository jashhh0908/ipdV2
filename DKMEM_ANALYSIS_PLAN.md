# DK-Mem analysis plan for Tier 1 v2 (pre-registered before any valid real-model run)

Written 2026-10-10, on top of commit `b2067be`, by the Team A maintainer who inherited the project. At this point **no valid
real-model result exists** (the earlier `results/abcd_3b_s0` run predates the `lexicon+llm` fix and is void, see
`DKMEM_BASELINES.md` §7), and **nobody on Team A has seen any gold label**. Everything below was fixed before the data
can be scored, so none of it is tuned on test outcomes. Team B can no longer answer questions or supply data, so
every decision Team B was meant to sign off is taken here and documented as a Team A decision.

## 1. Question and the claim we can defend

The original question was: does the distinction-aware merge gate lower false consolidation (FCR) of different
entities while keeping missed consolidation (MCR) acceptable?

Tier 1 v2 can only partly answer it:

- Gold is **LLM-annotated, not human-validated**. Every FCR/MCR number is labelled "Tier 1 v2, LLM-annotated, provisional".
- Class and relation are confounded. Kinship has 32 different / 0 same pairs. `honorific_register` is all *same* in Hindi
  and all *different* in German. Four of the seven classes are out of the gate's scope.
- The lexicon is Hindi only (12 kinship terms plus tu/tum/aap), so on `de`, `ko` and `tr` the gate is inert.

So the claim we test is narrowed to this:

> **On the same host decisions, does the `lexicon` gate lower FCR relative to gate-off, and how much MCR does it add,
> on Hindi Tier 1 v2 pairs, with labels from an LLM annotator?**

We do not claim cross-lingual generalisation, and we do not claim validated-benchmark performance. MCR is not
measurable on kinship (n = 0 same pairs), so any "MCR stays acceptable" statement rests on the other classes. These
are mostly out of scope for the gate: the gate's MCR cost there is spillover from incidental lexicon hits.

## 2. Analyses (fixed now)

**Primary.** This is the frozen design, unchanged: `DKMEM_FIRST_RUN.md` §1, gate features kinship + register.
- For each host config A, B, C and D (D at its run τ), compare mode `lexicon` with `off` on the same pairs.
- Report FCR and MCR with Wilson 95% intervals and the paired exact McNemar test (`ablation`, `sensitivity`).
- The four primary FCR tests are corrected with Holm at α = 0.05; all p-values are shown raw and adjusted.
- No single overall number is headlined (contract §8). Per-class and per-language tables, with n, come first.

**Secondary.**
1. `lexicon+llm` vs `lexicon` (the ablation). Expected to be nearly identical (the model is consulted only for
   lexicon-ambiguous utterances); report `extra_llm_calls` and the changed pairs.
2. Config D: the full raise-τ frontier vs DK-Mem on D (`sweep`, `compare`), FCR at matched MCR.
3. Baselines (all run, all reported, including non-informative ones):
   - flat RAG never merges, so its FCR = 0 and MCR = 1 by construction. It is an anchor, not a competitor.
   - The prompt-informed judge.
   - Mem0 OSS v1.0.11, reported with its failure counts.

**Pre-registered sensitivity (register).** Register (tu/tum/aap) marks how the speaker addresses the listener, not who
the fact is about, and the spec's gate condition is a feature that "discriminates identity"
(`DKMEM_NEW_RESEARCH_IDEA.md` §5.4). The codebase already keeps `name_variant` non-gating for the same reason: it has
same-entity controls. Register has them too.
- `python -m dkmem.eval.cli sensitivity` replays the logged gate offline with **kinship only** and reports it next to
  the frozen gate. It uses the same host decisions and extractions, and runs a paired McNemar test against both
  gate-off and the frozen gate.
- It also reports every rate over all pairs **except** `honorific_register` (13 pairs, all n < 20).
- The frozen gate stays the primary result whichever way this comes out. Register is neither dropped nor changed in code.

Known without the key, from the lexicon on the real utterances (fake always-merge host, 2026-10-10 smoke): in `lexicon`
mode the gate can veto at most 37 of 173 pairs. 32 of those are caused by kinship and **5 by register alone**. So the
register rule can move MCR by at most 5/75 ≈ 6.7 points, or FCR by at most 5/98 ≈ 5.1 points, in that mode.

**Key-free diagnostics.** `python -m dkmem.eval.cli diagnostics` needs no key. It gives aggregate counts only:
- the outcome mix, overall and per language;
- host merge proposals and vetoes;
- the veto cause (kinship / register / both);
- vetoes under kinship-only.

These are what Team A can report if the key never becomes available.

## 3. Deviations from the contract (Team A decisions; Team B sign-off is impossible)

| Contract / doc item | Decision | Reason |
|---|---|---|
| mean ± sd over ≥3 seeds (contract §8) | One seed (0), reported as a deterministic run. No sd. Uncertainty comes from Wilson intervals and paired McNemar tests. | Decoding is greedy with a fixed batch order, so seeds 1 and 2 reproduce seed 0 exactly. An sd of 0 would be misleading. |
| `threshold: null` for A–C, strategy labels, nullable fields (FIRST_RUN §6.3–4) | Kept as implemented. | Already schema-valid; changing them now would only break comparability. |
| Config D τ 0.80 / 0.85 / 0.90 | Kept as placeholders. | The `sweep` recomputes the whole τ curve offline from the logged scores, so the run τ affects only the single reported D point, not the frontier. |
| bge-m3 revision unpinned in FIRST_RUN | Pinned to `5617a9f61b028005a4858fdac845db406aefb181`. | This is the revision `DKMEM_BASELINES.md` and `dkmem/store/cli.py` already use, so D and the baselines share one embedder. |
| 6 skipped tests replay `results/abcd_3b_s0` | That run is void; the tests stay skipped. | It predates the `lexicon+llm` fix. |

## 4. Code changes made with this plan (2026-10-10)

- `dkmem/eval/sensitivity.py`, plus the `sensitivity` and `diagnostics` commands in `dkmem/eval/cli.py`.
  - The replay first re-applies the frozen gate and must reproduce every logged outcome; otherwise it refuses.
  - It accepts A–D groups only.
- `--gold` is required by every eval command except `diagnostics`. `--out` refuses to overwrite an existing file.
- `ensure_fresh_group_dir` (`dkmem/pipeline/runner.py`) is used by the A–D, Mem0 and flat writers.
  - A run never overwrites an existing group; it fails instead (previously the second run replaced the first silently).
- `pair_outcomes` rejects an `entry_id` that two utterances share (contract §3; previously it was not checked).
- `tests/test_key_protection.py` scans `results/`, `runs/`, `outputs/` and `validation/` (JSON and JSONL) for gold labels.
  Before, it scanned only `results/*.jsonl`.
- New tests: `tests/test_sensitivity.py` (13 tests). One `test_eval.py` test now writes its second group to a fresh folder.
- Not changed: gate, scope, lexicon, prompts, pinned models, input data. The `dkmem/` tree hash changes, so new
  group fingerprints differ from the void earlier run's. That is intended.

## 5. Commands

Local, no GPU: `python -m unittest discover -s tests`.

Real runs go through Kaggle (this machine has no CUDA). Use **your own** Kaggle account in place of `<USER>`; the
earlier docs name the previous owner's account.

```
M="--model-id Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1"
E="--embedding-revision 5617a9f61b028005a4858fdac845db406aefb181"

# 1. guard run, 10 records, every untested piece (bge-m3, CLI, judge on B)
python validation/abcd_first_run/build_kernel.py --user <USER> --out <scratch>/k_guard --slug dk-mem-abcd-3b-guard $M $E --seed 0 --tau-d 0.85 --limit 10
kaggle kernels push -p <scratch>/k_guard --accelerator NvidiaTeslaT4
kaggle kernels output <USER>/dk-mem-abcd-3b-guard -p results/guard_3b_s0_<date>

# 2. full A-D run (only after the guard's checks in DKMEM_FIRST_RUN.md §3 pass)
python validation/abcd_first_run/build_kernel.py --user <USER> --out <scratch>/k_full --slug dk-mem-abcd-3b-s0 $M $E --seed 0 --tau-d 0.80 0.85 0.90
kaggle kernels push -p <scratch>/k_full --accelerator NvidiaTeslaT4
kaggle kernels output <USER>/dk-mem-abcd-3b-s0 -p results/abcd_3b_s0_<date>

# 3. baselines: the same builder with --judge-prompt merge_judge_informed_v1 --configs A B C --modes off, --baseline mem0, --baseline flat
#    (DKMEM_BASELINES.md section 4), each to its own results/<name>_<date> directory

# 4. offline, key-free (Team A)
python -m dkmem.eval.cli diagnostics --group results/abcd_3b_s0_<date>/runs/* --out results/diagnostics_<date>.json

# 5. offline, needs the key (whoever holds it); --out never overwrites
K=--gold <path>/teamB_answer_key.jsonl
python -m dkmem.eval.cli ablation    --group <A> <B> <C> <D...> $K --out results/ablation_<date>.json
python -m dkmem.eval.cli sensitivity --group <A> <B> <C> <D...> $K --out results/sensitivity_<date>.json
python -m dkmem.eval.cli compare     --group <A> <B> <C> <mem0> <flat> <informed...> --d-group <D@0.85> $K --out results/compare_<date>.json
python -m dkmem.eval.cli sweep       --group <D...> $K --out results/sweep_<date>.json
```

`results/` is gitignored. Do not commit run outputs that sit next to a key, and never copy the key into `results/`.

## 6. What the results will and will not support

- **Supported, if the runs complete:** stage attribution and gate activity (key-free); paired gate-vs-off differences on
  Tier 1 v2 under LLM labels, per class, for Hindi.
- **Provisional:** every FCR/MCR number (LLM labels); every cell with n < 20, which means everything outside Hindi and all
  of `honorific_register`; the register sensitivity.
- **Not supported by this data:** MCR of the kinship gate (no same-entity kinship pairs); any effect in German, Korean or
  Turkish (no lexicon entries); generalisation beyond the 173 pairs; human-validated accuracy; seed variance.
