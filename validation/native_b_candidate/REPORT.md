# Candidate Config B prompt (native-language extraction) vs the current B prompt

**Run:** Kaggle notebook `jashnikumbhe/dk-mem-native-b-candidate`, version 1 (pushed = saved, then run), private, Internet on, "GPU T4 x2" (2 x Tesla T4 listed, GPU 0 used). Qwen/Qwen2.5-3B-Instruct revision `aa8e7253...`, fp16, greedy, seed 0, batch 8, max_new_tokens 256; merge judge `merge_judge_v1` (max_new_tokens 128); gate `apply_gate` with the lexicon; language report `output_language_report`. Same 20 records as the A-C smoke test (slice sha256 `fb2fd613...`). Nothing in `dkmem/` was modified; the candidate prompt is `candidate_prompts.py` in this folder (sha256 `6ca296ec...`, **not frozen**). The re-run of the current prompt reproduced the smoke-test raw outputs exactly (40/40), so the two runs are directly comparable.

**Candidate prompt:** one utterance -> `{"fact": string}`; exactly one short sentence; Mem0's own language sentence verbatim ("detect the language of the user input and record the fact in the same language") plus: keep language and script, do not translate, keep code-mixing, keep names and relationship terms exactly as written, add nothing, never return an empty fact. Zero examples. Strict parsing (no fence stripping).

## Results (40 sides, 20 pairs)

| | Current B (Mem0 legacy prompt) | Candidate B |
|---|---|---|
| Parsed | 40/40 | 40/40 |
| Sides with exactly one fact | 14/40 (26 empty) | **40/40** |
| Pairs reaching the judge | 3/20 | **20/20** |
| Judge answers valid | 3/3 | 20/20 |
| Language label (heuristic) | 10 translated, 4 mixed, **0 source** | **34 source**, 4 translated, 2 mixed |
| Mean share of non-English words kept | 0.20 | 0.86 |
| Devanagari utterances that stay Devanagari | 1/3 | **3/3** |
| Verbatim copy of the utterance | 0 | 26/40 (65%); 30/40 near-copy (>= 0.9) |
| Relationship terms kept (*mami, mausi, Dadi, Nani, uncle, bhaiya*, 8 sides) | n/a (mostly no fact) | 8/8 |

## Language drift and information loss in the candidate (read by hand)

The heuristic undercounts drift: a name that survives counts as a kept word, so English or Spanish sentences containing only a name are labelled "mixed". Reading all 40 outputs:

- **Drifted to another language: 6/40 (15%)** - 5 English ("Rohan gave one carton of milk.", "Rohan gave some strange milk.", "Manoj gave some strange milk.", "Suresh gave two clothes to him.", "Anjali is coming.") and 1 Spanish ("Neha está llegando."). Four of the six sit on utterances with rarer Latin-script Hindi nouns or quantity words (*botal, gilaas, jode, tukde*), where the 3B model seems not to understand the word and paraphrases in English.
- **Native but damaged: 4/40** - an invented relationship term ("Manoj **dadaa** ek botal doodh laaya"), garbled words ("Kal Manoj Pune **gareebega**", "Vikram **paī** botal paani piya"), and a dropped demonstrative ("**laptop** mera hai" for "woh laptop mera hai").
- **Faithful: 30/40**, of which 26 are exact copies; the rest differ only by a trailing full stop.
- Information loss that matters for merging: "gave two clothes" loses *jode* vs *tukde*; the invented *dadaa* adds a kin term that was never said. Current B's losses were larger: "carton", "Laptop is mine", "अंजलि is awake" (invented), tense changes, and the Devanagari script dropped.

## Downstream (unchanged judge and gate, candidate B)

The judge kept 9 pairs apart and merged 11, all with valid JSON. The one lexicon-detectable kinship conflict (*mami* vs *mausi*) stayed intact in both stored facts and the judge kept it apart. The judge merged, among others, *gaya tha* vs *jayega* (past vs future), *uncle* vs *bhaiya*, and *dein* vs *Ihr* (German register). This slice cannot say anything about error rates; it only shows the pipeline now sees every pair.

## Assessment

**Does it satisfy the requirements?** Yes: exactly one fact per utterance (40/40), original language and script kept on 34/40 sides (85%), names and relationship terms preserved on all kinship sides, no empty outputs, no translation instruction violated by design. **It fixes both defects of the current B** (3/20 pairs reaching the judge; no source-language facts).

**Does it test L2?** Yes, and the result is informative: told clearly to keep the language, the 3B model still drifts on about 15% of sides, concentrated where it does not understand the word. That is the L2 phenomenon, measured under a clear instruction.

**Caveats before freezing:**
1. **Most outputs are copies.** 65% are verbatim, 75% near-verbatim, so on those sides B is the same as Config C. That is legitimate (no drift), but the B-vs-C contrast will come almost entirely from the ~25% of sides that differ. Report the copy rate beside the drift rate.
2. **The language classifier needs a fix.** Names inflate the "words kept" measure (the "mixed" bucket is really English/Spanish). Exclude names or add an English-function-word share, and hand-check a sample, before a drift rate goes in the paper.
3. **Small sample.** 20 pairs, one seed, greedy decoding, mostly legacy-scope pairs; German, Turkish and Devanagari appear 2-6 times each. The 15% drift is an estimate with a wide interval.
4. **Model-quality noise.** 4 corrupted outputs (including an invented kin term) are extraction errors, not language drift; keep them as a separate category so they are not read as L2.

## Recommendation

**Adopt this prompt (after the checks above) as Config B**, replacing the Mem0 legacy prompt, and keep real Mem0 for the baselines. Suggested order before freezing: (1) fix the language classifier and hand-check ~30 outputs; (2) run the candidate on the 69 in-scope Tier 1 pairs (or all 173) to get stable drift, copy and corruption rates, and on 1.5B/7B to see the size trend (H3); (3) only then freeze the prompt text and give it an id. Do not add examples or a "rewrite, don't copy" instruction now: examples would suppress the very drift being measured, and forcing a rewrite would add paraphrase damage.

## Files

`candidate_prompts.py` (the candidate, unfrozen), `build_kernel.py`, `analyze.py`, `metrics.json`, `side_by_side.md` (all 40 sides and 20 pairs, current vs candidate), `kaggle_output/native_b_comparison.json` (raw model outputs, judge outputs, run environment), Kaggle log.
