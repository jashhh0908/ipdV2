# Candidate Config B prompt: pre-freeze checks (173 Tier 1 records x Qwen2.5 1.5B / 3B / 7B)

**Run:** Kaggle notebook `jashnikumbhe/dk-mem-native-b-prefreeze`, version 1 (pushed = saved, then run), private, Internet on, "GPU T4 x2". Candidate prompt `native_b_candidate_v1` (sha256 `6ca296ec...`, **unmodified, not frozen**; text in `../native_b_candidate/candidate_prompts.py`). All 173 sanctioned Tier 1 records (file sha256 `eab5dcb2...`), both sides = 346 prompts per model. Qwen2.5-1.5B-Instruct (rev `989aa798`), 3B (`aa8e7253`), 7B (`a09a3545`); fp16; greedy, seed 0, batch 8, max_new_tokens 256 (identical for the three). 1.5B and 3B ran on one T4; the 7B does not fit one T4 in fp16, so its fp16 weights were sharded over both T4s (`device_map`; same dtype and decoding). No merge judge, no A/C/D, no baselines. 0 errors.

## 1. Classifier fix and validation

**Fix** (`dkmem/memory/native_extract.py::output_language_report`, now `classifier_version: "v2"`): the retention measure no longer counts names, loanwords or cross-language words as words kept from the source. It is computed over *content words* only: capitalized words (names, and German nouns) are dropped, as are the English loanword list, a short list of words spelled the same in English and the data's languages (Hindi *the* = "were", German *in/was/war*), and non-Latin tokens inside a Latin utterance. A second signal, the share of unambiguous English function words, separates "mixed" from "source". A new label `script_changed` marks a Latin-script utterance rewritten in another script. 4 new unit tests (suite: 459 pass, 1 skipped).

**Validation:** every output that is not a copy of its utterance (case, punctuation and spacing ignored) was read by hand, blind to the classifier's label: 552 unique (utterance, output) pairs covering 603 non-copy sides; copies are source-language by definition. Each got a language label (S source and script kept; E English; O other Latin-script language; T same language, different script; M mixed; N other-script/other-language or unintelligible) and a corruption label (below). `hand_labels.txt` has all of them.

| | v1 (old) | v2 (fixed) |
|---|---|---|
| Agreement with hand labels, non-copy sides (n=603) | 51.2% | **93.4%** |
| Agreement, all sides (n=1038) | 71.7% | **96.1%** |
| Hand-translated sides called "mixed" (the name loophole) | 98 | 18 |
| Per model, non-copy: 1.5B / 3B / 7B | 62.5 / 77.6 / 16.7% | 93.2 / 98.0 / 91.1% |

Rate check (sides labelled translated, hand vs classifier): 1.5B 319 vs v2 297 (v1 198); 3B 30 vs 30 (v1 20); 7B 1 vs 1 (**v1 146**). The 7B number is why the fix mattered: v1 called its 143 Devanagari rewrites "translated to English". "Not source-language" (any departure): hand 321 / 41 / 152 vs v2 317 / 43 / 159.

**Remaining v2 errors (40 sides):** 18 hand-translated sides called mixed (an English sentence that keeps a Hindi relationship term, or an unlisted English loanword such as *bag*, *scooter*); 13 source-language sides called mixed (an output that drops words such as *suna hai*, which lowers retention); 9 other. The classifier is reliable for the rate of departure from the source and for script change, less so for the source/mixed boundary. I did **not** tune it further on these labels (that would be in-sample).

**Limits of the validation:** one annotator (me), blind to the classifier, not to the task; borderline conventions (an English sentence that keeps a kin term is labelled E here; the classifier says mixed). The `script_changed` rule was added after I saw the Devanagari outputs, so it is in-sample; the thresholds and word lists were fixed before the new data.

## 2. Results (hand-checked unless stated; 346 sides per model; 95% Wilson CI treats sides as independent, which they are not quite, since pairs are minimal pairs)

| | 1.5B | 3B | 7B |
|---|---|---|---|
| Parsed | 346/346 | 346/346 | 346/346 |
| Exactly one fact (one sentence) | 344/346 (99.4%) | 346/346 | 346/346 |
| **Language drift** (translated to English/other language) | **319 (92.2%)** [89-95] | **30 (8.7%)** [6-12] | **3 (0.9%)** [0-2] |
| of which English | 319 | 24 | 1 |
| **Script drift** (same language, other script, e.g. Hinglish -> Devanagari) | 1 (0.3%) | 9 (2.6%) | **143 (41.3%)** [36-47] |
| Mixed (source + English in one fact) | 1 | 2 | 6 |
| Source language and script kept | 25 (7.2%) | 305 (88.2%) | 194 (56.1%) |
| Any departure from the instruction | 321 (92.8%) | 41 (11.8%) | 152 (43.9%) |
| **Extraction corruption** (any) | **257 (74.3%)** [69-79] | **62 (17.9%)** [14-22] | **81 (23.4%)** [19-28] |
| corruption excluding "dropped a word" | 222 (64.2%) | 44 (12.7%) | 43 (12.4%) |
| corruption types: garbled / invented / meaning changed / word dropped | 2 / 1 / 220 / 56 | 22 / 1 / 36 / 20 | 39 / 1 / 8 / 41 |
| **Copy rate** (exact / ignoring case, punctuation, spaces) | 17 (4.9%) / 21 (6.1%) | 190 (54.9%) / 248 (71.7%) | 160 (46.2%) / 166 (48.0%) |
| **Kinship terms kept as written** (sides containing one, n=118) | 36 (30.5%) | 115 (97.5%) | 97 (82.2%) |
| of which lexicon-listed Hindi kin terms (n=76) | 15 (19.7%) | 75 (98.7%) | 58 (76.3%) |
| Kin terms lost because | translated 81 | translated 2, garbled 1 | garbled 20, translated 1 |
| Classifier v2 (source / translated / mixed / script_changed) | 29 / 297 / 19 / 1 | 303 / 30 / 4 / 9 | 187 / 1 / 12 / 146 |

**Drift and corruption are separate.** Corruption is content damage in any language: garbled non-words, invented content, changed meaning (wrong tense, object, subject, question made a statement), or a dropped content word. Cross-tab (drift x corruption): 1.5B: 254 of 319 drifted sides are also corrupt; 3B: 19 of 30; 7B: 3 of 3, plus 44 of the 143 script-changed sides (Devanagari misspellings). 3B has 37 corrupt sides that stayed in the source language and script.

**By language (3B):** hi 26 language drifts of 290 sides, ko 4 of 24, de 0 of 12, tr 0 of 20. 1.5B drifts on essentially every language (de 11/12, ko 21/24, tr 18/20, hi 269/290). 7B keeps ko/de/tr entirely in source and only rewrites Hinglish in Devanagari.

**Examples**
- *1.5B, drift + corruption:* "Rohan ek botal paani piya" -> "Rohan has a water bottle."; "Deepak ne do jode kapde diye" -> "Deepak is a male name."; "Nani ji ne achaar bheja" -> "The user wants to know about a dish called 'bheja gosht'."
- *1.5B, kin lost by translation:* "meri chachi Pune mein rehti hai" -> "My aunt lives in Pune." but "meri mausi Pune mein rehti hai" -> "My mother is in Pune." (kin terms kept as written in only 30.5% of kin sides); Korean "딸이 결혼했어요" -> "The daughter got married."
- *3B, drift:* "Neha aa rahi hai" -> "Neha está llegando." (Spanish); "Suresh ne do tukde kapda diya" -> "Suresh gave two clothes to him."; Korean "아내가 병원에 갔어요" -> a Chinese-script sentence.
- *3B, corruption in the source language:* "kal Manoj Pune jayega" -> "Kal Manoj Pune gareebega."; "kal Rohan Jaipur jayega" -> "Kal Rohan jaigaya."; "Dadi ji ne kahani sunayi" -> "Dadi ji kahani sunaya" (dropped *ne*).
- *7B, script drift:* "Anjali aa rahi hai" -> "अंजाली आ रही है"; "meri chachi Pune mein rehti hai" -> "मेरी चाची पुणे में रहती है" (script changed, kin term intact); "meri mausi Pune mein rehti hai" -> "मेरी मा�usi पुणे में रहती है" (script changed and the kin term garbled).
- *Other:* 1.5B "Can geldi" -> "Geldi is Turkish for 'it is true'."; 7B leaked the prompt label once ("Utterance: Can geldi") and wrote several Hindi sentences in Urdu script, some garbled; 3B wrote a Hebrew/Chinese mixture and a Chinese sentence for two Korean sides.

## 3. What the numbers say

- **The prompt works as a measurement instrument:** 100% parse, >= 99.4% exactly one fact, no empty outputs, on all three models.
- **L2 is real but not monotonic in size.** Language drift: 92% (1.5B) -> 9% (3B) -> 1% (7B), which supports H3 between 1.5B and the larger models. The 7B instead breaks the *script* instruction on 41% of sides. So "small models drift into English" holds for 1.5B only; for 7B the failure is a different one.
- **At 1.5B, B is effectively English-forced extraction** (92% English, 74% corrupt), so it largely reproduces L1 conditions plus heavy corruption. B at 1.5B will say more about the extractor's competence than about the user's-language instruction.
- **Native-language storage preserves kin terms only if the model keeps them:** 97.5% (3B), 82.2% (7B; losses are garbled Devanagari), 30.5% (1.5B).
- **B is mostly C at 3B** (72% copies), less so at 7B (48%) and not at 1.5B (6%).
- **Corruption is not small:** 12-13% of sides at 3B/7B even excluding dropped words, and it is concentrated in garbled Devanagari/Latin spellings and in tense/meaning changes. It must be reported next to drift and will affect the judge in any B run.

## 4. Recommendation

**Freeze the candidate as Config B, with the prompt text unchanged**, on these conditions:
1. **Do not tune the prompt toward better behaviour.** Drift, script drift and corruption are what B measures; adding examples or a "rewrite, don't copy" clause would change the phenomenon (and the 7B/1.5B results are informative as they are).
2. **Report four separate B outcomes per model:** language drift, script drift, mixed, and corruption (plus the copy rate). Do not collapse script change into "translated"; it is a different failure that mostly preserves the distinction (and the lexicon has Devanagari forms) while breaking the verbatim-script contract.
3. **Adopt classifier v2 as the automatic instrument** (agreement 93% on non-copy sides; population-level departure rates within about 2 points), and keep a second annotator on a random 100-output subset to report agreement; the current labels are single-annotator.
4. **Treat B at 1.5B as a near-L1 condition** in the write-up, and note the 7B script behaviour as a result, not a bug.
5. Open items that do not block freezing: the English-frame-plus-kin-term convention (E vs mixed), 7B sharding across two GPUs (numerics may differ slightly from single-GPU runs), one seed with greedy decoding, and Tier 1 being very short templated sentences (the drift rates may not carry over to longer text).

If you would rather have B reflect a regime where drift is small (a strictly "native" condition), that is a different design choice (a stronger prompt or examples); I would run it as an additional condition, not as a replacement for this one.

## Files

`kaggle_output/native_b_<model>.json` (raw outputs, run info), `parsed.json`, `to_label.jsonl` + `hand_labels.txt` (all 552 hand labels: `id lang corruption kin`), `labeled_sides.json` (every side with hand and classifier labels), `metrics.json` (all numbers above, confusion matrices, the 40 classifier disagreements), `analysis.py` / `report_stage.py` (reproduce everything), `build_kernel.py`. Code change: `dkmem/memory/native_extract.py` (classifier v2) and `tests/test_pipeline.py`.
