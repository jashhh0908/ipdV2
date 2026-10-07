# Prompt V4 extraction validation: V2 vs V3 vs V4

**Run:** Kaggle notebook `jashnikumbhe/dk-mem-prompt-v4-validation`, version 1 (saved and run), Internet on, accelerator "GPU T4 x2" (`nvidia-smi -L` listed 2 x Tesla T4; decoding used GPU 0 only, as in the earlier V1-V3 runs). Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, fp16, greedy, seed 0, batch size 8, max_new_tokens 256, torch 2.11.0+cu128, transformers 5.16.1.
**Inputs:** the same 20 sides as before (the 10 synthetic probes in `tests/fixtures/probe_items.jsonl`, sides a and b), same order, one fresh cache per prompt. V2, V3 and V4 ran in one kernel on one loaded model, with identical inputs and scoring.
**Check that the setup matches the earlier runs:** V2 reproduced 10/20 target-feature and 9/20 exact; V3 reproduced 10/20 and 6/20 (the numbers from the earlier V1-V3 comparison). The 6 embedded unit-test files passed on Kaggle (`OK`). V1-V3 were not modified (their fingerprints are checked by the tests).

## Results

| Metric | V2 | V3 | V4 |
|---|---|---|---|
| Correct distinctions (target-feature), of 20 | 10/20 | 10/20 | 8/20 |
| Exact-match distinctions, of 20 | 9/20 | 6/20 | 8/20 |
| **Correct in-scope distinctions** (kinship / register / name_variant), of 12 | 7/12 | 7/12 | **8/12** |
| In-scope exact match, of 12 | 7/12 | 6/12 | 8/12 |
| Correct on the 8 out-of-scope probe sides | 3/8 | 3/8 | 0/8 |
| Wrong value (expected key present, wrong value) | 3 | 4 | 2 |
| Missing (expected key absent) | 6 | 5 | 9 |
| Over-marked (unmarked side given a key) | 1 | 1 | 1 |
| Invalid / extraction failure | 0 | 0 | 0 |
| Spurious keys in valid outputs (any key not expected) | 10 | 12 | **6** |
| of which out-of-scope class keys | 2 | 5 | **0** |
| of which in-scope keys (wrong place / unmarked side) | 8 | 7 | 6 |
| Out-of-scope keys in rejected outputs | 0 | 0 | 0 |
| `you` / `My...` gloss violations | 4 | **0** | 1 |
| Invalid JSON / extraction failures | 0 | 0 | 0 |
| Surface mismatches | 0 | 0 | 0 |

Criteria are those of the earlier comparison: *target-feature correct* = the expected `{feature: value}` is present (extra keys ignored; for the unmarked side, the feature is absent); *exact* = distinction equals the expected one. In-scope = probe classes kinship, register, name_variant, control_kinship (12 sides). Spurious = a key in a valid output that is not the expected key. Gloss violation = whole word I / me / my / you in the gloss. Wrong/missing/over-marked split the sides that are not correct. In-scope breakdown (correct / wrong / missing / over-marked): V2 7/2/2/1, V3 7/3/1/1, V4 8/2/1/1. Full numbers: `metrics.json`; every raw model output: `kaggle_output/extraction_validation_*.json` and `extraction_cache_*.jsonl`; side-by-side items and glosses: `per_item_comparison.md`.

## What the numbers mean

- **The headline drop (10/20 to 8/20) is a scope effect, not a regression.** V4 cannot produce evidentiality, classifier, politeness or temporal_deixis, so it scores 0/8 on the sides that target them (V2 and V3 get 3/8). Nothing in scope got worse.
- **On the 12 in-scope sides V4 is at least as good: 8/12 vs 7/12.** The whole gain is one side (probe_005/a: `Priya-latin` was `Priya-devanagari` in V2 and V3). One item on a dev set is not evidence of improvement.
- **Spurious keys halved (12 to 6) and none are out-of-scope classes** (V3 emitted 5, V2 emitted 2). V4's strict key rejection never triggered: no output was rejected, so there were no extraction failures.
- **One new gloss violation:** probe_003/a, "you come to the office tomorrow" (V3 wrote "addressee will come ..."). V3 remains the cleanest on this rule (0), V2 the worst (4).
- **Unchanged in-scope failure modes** (same in V2, V3 and V4): *chachi* is glossed "grandmother" on 4 of 4 sides (a model-knowledge error the prompt does not fix), *aunty* is tagged as kinship (probe_004/b), name script is misjudged (probe_005; V4 also invented the typo `Priva-devanagari`), German *Bruder* stays untranslated, and a spurious `register` appears on German sentences without *du* (`du` in V2/V3, `meine` in V4).
- **Some "spurious" keys are arguably correct:** `name_variant: Ali-latin` on probe_006 names a real person, but the probe only labels the evidentiality feature, so the criteria count it (V4 has 2 of these, and 2 `register: 선생님` that are plainly wrong).

## Caveats

- Same 20 sides used to design V2-V4, so all numbers are development-set and optimistic. 12 in-scope sides is too few for any significance claim; greedy decoding means each cell is one deterministic run.
- Only 4 of the 12 in-scope sides test name_variant/register, and German/Korean/Turkish sides are single examples.
- In the DK-Mem pipeline the lexicon overrides the model's kinship/register values wherever it matches (all of the kinship/register probes here are in the lexicon), so these model-only numbers matter most for the fallback path and for sentences the lexicon does not cover.

## Conclusion: should V4 replace V3 for the new experiments?

**Yes, adopt V4 as the extraction prompt, with the caveat that it is a scope-consistency decision rather than a demonstrated accuracy gain.**
- V4 is the only version whose allowed keys match the new scope, the gate's cannot-link set and the Tier 1 schema; V3 emitted out-of-scope keys on 5 sides, which the new pipeline would have to reject or ignore.
- On the in-scope sides it is not worse (8/12 vs 7/12, exact 8/12 vs 6/12) and it halves spurious keys, which matters because a spurious `kinship` or `register` value can create a false `incompatible` or `underdetermined` outcome at the gate.
- Costs: one gloss-rule regression (1 violation vs 0 for V3), unchanged weaknesses (*chachi* gloss, name script, *aunty*), and nothing about it improves the in-scope kinship/register values the lexicon already supplies.
- Do not describe V4 as more accurate than V3 on this evidence. The in-scope difference is one item. Keep V3 and V2 as frozen comparison prompts, as already done.

## Files

- `build_kernel.py` builds the notebook (embeds only source and synthetic fixtures); `score.py` computes all metrics; `metrics.json`, `per_item_comparison.md`; raw outputs, caches, run environment and the Kaggle log in `kaggle_output/`.
