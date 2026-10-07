# DK-Mem: Language-Induced False Consolidation in LLM Agent Memory

**Working title:** *Where the Distinction Dies: Language-Induced False Consolidation in LLM Agent Memory*

**Target:** ≤6-page paper, small/mid-tier IEEE workshop
**Team:** four undergraduates, two execution teams (Team A: Pipeline & Measurement; Team B: Data, Linguistics & Evidence)
**Compute:** free Kaggle / Colab GPU only, no training
**Timeline:** 30-day plan with gates G1–G4

---

## 1. Problem and core phenomenon

LLM agent memory systems such as Mem0 and A-Mem do two things with what a user says. They **extract** facts from it, and they **consolidate** those facts over time. Consolidation means deciding whether a new fact is a duplicate of, an update to, or separate from something already stored. When the system decides two facts refer to the same thing, it merges them or lets the newer one supersede the older one.

That decision is only as good as the evidence that reaches it.

Multilingual users often make distinctions that English does not have a single word for. In Hindi, *chachi* (father's brother's wife), *mami* (mother's brother's wife), *bua* (father's sister) and *mausi* (mother's sister) are different people. In English, all of them are "aunt."

Consider a user who says, in different sessions:

- "meri chachi Pune mein rehti hai" (my chachi lives in Pune)
- "meri mausi Delhi mein rehti hai" (my mausi lives in Delhi)

If anything in the memory pipeline reduces both to "user's aunt lives in X," the consolidation step can treat the second fact as an **update** of the first ("aunt moved from Pune to Delhi"). The Pune fact is then deleted. No retrieval technique can recover it, because it is no longer in the store. The system reports the wrong answer with full confidence.

**We call this language-induced false consolidation:** the wrongful merging or supersession of two distinct facts whose only distinguishing evidence is a source-language lexical distinction.

This is a **memory-integrity** failure, not just a translation-quality failure. It destroys a true fact, not merely the answer to a question about it.

A deployed example shows the risk is real. Hindsight, a commercial memory system, stores facts in the user's language by default. But when one consolidated observation merges facts written in several languages, the majority language wins and the observation is rewritten in it. In other words, the system commits to a language *during merging*.

---

## 2. Final research question

> When a multilingual user states two facts that differ only in a source-language distinction, **at which stage of a memory pipeline does that distinction stop influencing the merge decision**, and does an explicit **distinction key**, used as a cannot-link constraint at merge time, prevent false consolidation where threshold tuning, prompting, and verbatim storage do not?

---

## 3. Hypothesis

**H1. False consolidation occurs.** Representative memory pipelines (Mem0, A-Mem) false-merge a substantial fraction of minimal pairs that are distinct in Hindi/Hinglish but identical after English normalization.

**H2. Keeping the original language is not sufficient.** False consolidation persists even when facts are stored natively or verbatim. This is because the merge judge (an LLM with an English prompt) or the embedding-based candidate selection can still collapse fine-grained distinctions such as *chachi* vs *mausi*.

**H3. Small models make it worse.** False consolidation rises as the backbone shrinks (Qwen2.5-7B → 3B → 1.5B), partly because small extractors drift into English despite instructions to keep the user's language.

**H4. An explicit distinction key fixes it at low cost.** A lexicon-derived distinction key, applied as a cannot-link constraint at merge time, lowers the False Consolidation Rate at matched Missed Consolidation Rate. It does better than raising the merge threshold, better than telling the merge judge about distinctions in its prompt, and better than verbatim storage.

If H2 is not supported (that is, native and verbatim storage both show low false consolidation), the paper narrows to a single claim: *English-forced extraction and consolidation are merge-unsafe for multilingual users*, and DK-Mem is evaluated in that setting only.

---

## 4. Where the distinction can be lost in the memory pipeline

The distinction can be lost at four points. The paper measures each one separately instead of assuming which one is responsible.

| # | Stage | How the distinction is lost |
|---|---|---|
| L1 | **Extraction: forced English** | The deployment or research prompt tells the extractor to write facts in English. *Chachi* becomes "aunt" before anything is stored. |
| L2 | **Extraction: language drift** | The prompt tells the model to keep the user's language (Mem0's default does this), but small models drift to English anyway. Code-mixed Hinglish makes "the user's language" ambiguous to begin with. |
| L3 | **Merge judge collapse** | Facts are stored with *chachi* and *mausi* intact, but the LLM that chooses ADD / UPDATE / DELETE treats both as "aunt" and calls the second an update. |
| L4 | **Embedding collapse** | Merge candidates are chosen by vector similarity, and the multilingual embedder places fine-grained same-category terms close enough to pass the merge threshold. |

L1 and L2 destroy the distinction **before storage**. L3 and L4 destroy it **at decision time**, even when it is stored. The stage analysis in Section 6 isolates these by varying extraction, storage, and merge mechanism independently.

---

## 5. Proposed distinction-aware solution: DK-Mem

### 5.1 Design principle

> Do not let a merge decision run on evidence from which the distinguishing feature has already been removed. Keep the distinction as structured data next to the memory, and treat a disagreement as grounds to refuse the merge.

### 5.2 Memory entry

Each memory entry stores three fields alongside the normal content:

```
{
  gloss:        "user's aunt lives in Pune",           # normalized text, used as retrieval key
  distinction:  {kin: "chachi"},                       # structured source-language distinction
  surface:      "meri chachi Pune mein rehti hai"      # verbatim original
}
```

### 5.3 Distinction extraction: lexicon first, model second

1. **Lexicon lookup.** A curated distinction lexicon is matched deterministically against the utterance. The kinship slice is generated from Kinbank, a typological kinship database, so the lexicon instantiates a principled category (distinctions a language lexicalizes and English does not) rather than an ad hoc word list.
2. **Model fallback.** A small LLM call is made **only** for spans the lexicon flags as ambiguous.

Both variants are evaluated: lexicon-only (zero extra inference) and lexicon + LLM.

### 5.4 Merge gating: `compatible()`

The merge rule changes from:

```
merge(a, b)  if  sim(a, b) > τ
```

to:

```
merge(a, b)  if  sim(a, b) > τ  AND  compatible(distinction_a, distinction_b)
```

`compatible()` is deterministic and has three outcomes:

- **Compatible.** Keys agree, or neither entry carries a key for that feature. The merge is allowed.
- **Incompatible.** Keys are different values of the same feature (*chachi* vs *mausi*). The merge is blocked and both entries are kept as separate entities. This is a **cannot-link constraint**.
- **Underdetermined.** One entry has a key and the other does not, for a feature that discriminates identity. The merge is blocked and the entries are **linked with an unresolved flag**, not merged and not duplicated blindly.

The gate is placed in front of whatever merge mechanism the host system uses, whether that is an LLM judge (Mem0, A-Mem) or an embedding threshold. It does not replace that mechanism.

### 5.5 Positioning of the mechanism

The constraint type is not new. It is the attribute-agreement constraint from deterministic coreference resolution (the Stanford multi-pass sieve, where dictionary-backed attributes such as gender and number block incorrect merges) and a cannot-link constraint from constrained clustering. What DK-Mem contributes is **the source of the attributes** (cross-linguistic lexical distinctions that the memory pipeline itself erases) and **the setting** (identity decisions during agent memory consolidation).

### 5.6 Cost

- Lexicon lookup: dictionary match.
- `compatible()`: dictionary comparison.
- Storage: one short structured field per entry, plus the verbatim surface text.
- Extra LLM calls: zero in the lexicon-only variant.

---

## 6. Experimental design and baselines

### 6.1 Data

**Tier 1: minimal-pair conflation probes (~200 items, core).**
Each item is two utterances that are **distinct in the source language and identical after English normalization**. Each has a gold label (same entity / different entity) and the distinction class responsible. Items are hand-authored by Team B and cross-validated by a second native speaker, with Cohen's κ reported. Phrasing is grounded in realistic Hinglish from LinCE.

Distinction classes in scope:

- **Kinship specificity** (primary class, generated from Kinbank)
- **Honorific / register** (secondary)
- **Name-variant surface form**, e.g. Priya / प्रिया / Preeya (secondary; includes same-entity pairs that should merge, which supports MCR measurement)

Each class includes **same-entity controls** (pairs that should merge) so both error directions are measured.

**Tier 2: injected multi-session conversations (~60 conversations).**
Tier 1 pairs are injected across distant sessions of LoCoMo multi-session scaffolds, which are already compatible with the Mem0 / A-Mem evaluation scripts. Three variables are controlled: input form (Hindi native script / romanized Hinglish / English control), session distance, and code-mixing ratio.

**Tier 3: non-Indic generalization slice (~30 items).**
One non-Indic language, using kinship distinctions drawn from Kinbank, so the finding is about normalization in memory pipelines and not a Hindi-specific quirk. The language is chosen by annotator availability. One language done well is preferred over several done thinly.

**Ecological validity check.**
Estimate how often the in-scope distinction classes occur in naturally code-mixed text (LinCE, and the IndicTalk code-mixed conversation corpus) to show the probe classes are not contrived.

### 6.2 Stage-attribution experiment (the core experiment)

Four pipeline configurations isolate the loss points from Section 4:

| Config | Extraction | Storage | Merge decision | Isolates |
|---|---|---|---|---|
| **A** | English-forced prompt | English gloss | LLM judge | L1 |
| **B** | Mem0 default prompt (record facts in the user's language) | Whatever the extractor emits | LLM judge | L2 |
| **C** | None (verbatim) | Surface text | LLM judge | L3 |
| **D** | None (verbatim) | Surface text | Embedding threshold (bge-m3) | L4 |

Each configuration is run with and without the DK-Mem gate, across Qwen2.5-1.5B / 3B / 7B. In Config B, we record **what language the extractor actually wrote**, which directly measures drift.

**Week-1 pilot.** 30 hand-made pairs across all four configurations, before full probe construction. This is the go/no-go check for H1 and H2.

### 6.3 Baselines

| Baseline | Why it is included |
|---|---|
| **Mem0, English-forced extraction** | Standard extract-and-consolidate pipeline under L1 conditions |
| **Mem0, default native-language prompt** | The realistic default; tests whether keeping the user's language is enough |
| **A-Mem** | Linked-note consolidation with a different merge logic |
| **Store-surface-only** | Verbatim storage with the host merge judge (Config C); answers "just don't normalize" |
| **Embedding-threshold merge** | Similarity-only merging (Config D); represents threshold-based merge policies |
| **Raise τ (conservative consolidation)** | The free, obvious fix; DK-Mem must beat it at matched MCR |
| **Prompt-informed merge judge** | One instruction added to the merge prompt: "treat different kinship terms as different people"; tests whether prompting alone closes the gap |
| **Flat dense RAG (bge-m3), no consolidation** | The no-merge control; FCR is zero by construction, so it anchors the cost side |
| **DK-Mem, lexicon-only** | Deterministic variant |
| **DK-Mem, lexicon + LLM** | Full variant |

Every consolidating baseline must be confirmed to actually perform merges before it is included. A system that never merges has FCR ≈ 0 trivially and does not test the phenomenon.

### 6.4 Ablations

- Lexicon-only vs lexicon + LLM
- FCR by distinction class (which classes are most dangerous)
- Extraction prompt language (English prompt vs native prompt)
- Backbone size (1.5B / 3B / 7B)
- Lexicon coverage: fraction of real distinctions caught, and performance on the uncovered remainder

---

## 7. Evaluation metrics

**Primary**

- **False Consolidation Rate (FCR):** fraction of distinct-entity pairs that the system merges or treats as a supersession.
- **Missed Consolidation Rate (MCR):** fraction of same-entity pairs left unmerged. This is reported alongside FCR so that "never merge" cannot look like a solution.
- **FCR–MCR frontier, swept over the similarity threshold τ.** This is the paper's main figure. The claim to establish is that DK-Mem achieves lower FCR than the baselines at matched MCR, including against simply raising τ.

**Stage attribution**

- FCR per configuration (A–D), with and without the gate
- Extractor output-language rate in Config B

**Secondary**

- **Downstream QA accuracy** on questions whose answers depend on a fact destroyed by false consolidation
- **Memory bloat:** stored entries per conversation, and count of unresolved links created by the underdetermined branch, plotted against FCR reduction (the integrity–growth trade-off)

**Statistics**

- Paired bootstrap over pairs, 95% confidence intervals, ≥3 seeds
- Cohen's κ on gold identity labels, reported for every annotated tier

---

## 8. Expected contributions and novelty

Stated as defensible paper claims. Claims 1 and 2 are conditional on the week-1 pilot.

1. **We identify and measure language-induced false consolidation in LLM agent memory:** the wrongful merging or supersession of distinct facts whose only distinguishing evidence is a source-language lexical distinction. To our knowledge, existing memory-integrity benchmarks (e.g., HaluMem, MOSAIC's conflict detection) are English-only and do not isolate this cause.

2. **We attribute the failure to pipeline stages.** We show how false consolidation changes when the distinction is removed at extraction, preserved in storage but collapsed by the merge judge, or collapsed by embedding-based candidate selection, including whether verbatim storage is sufficient to prevent it.

3. **We adapt attribute-agreement constraints from deterministic coreference resolution to agent memory consolidation,** using source-language distinctions that English does not lexicalize as cannot-link evidence, extracted by a lexicon-first cascade with a model fallback.

4. **We release a human-validated minimal-pair probe set** for merge decisions under cross-lingual and code-mixed input, annotated with gold identity labels, the responsible distinction class, and inter-annotator agreement, including a non-Indic slice.

5. **We quantify the integrity–growth trade-off of conservative merging,** comparing distinction-keyed blocking with threshold tuning, prompt-based instructions, and no-consolidation storage on FCR, MCR, and stored-entry count across a threshold sweep.

**Related-work boundaries the paper states explicitly**

- **MOSAIC** owns save-time conflict detection. It targets contradiction (same entity, conflicting values). DK-Mem targets false identity (different entities, same normalized form).
- **DPCA** owns the move of keeping a feature that single-stream extraction destroys (speaker identity) and measuring a dedicated conflation rate. DK-Mem applies the same move to a linguistic feature that the pipeline itself erases.
- **HaluMem** owns operation-level update error measurement. FCR is a cause-specific subset of update error.
- **THEANINE** and **Keep Me Updated!** already keep both memories instead of deleting. DK-Mem's contribution is doing so *conditionally* and measuring the cost.
- **TANGLE** argues agents should recognize underdetermination and preserve alternatives at answer time. DK-Mem applies that principle at write time.
- **The Late Filtering Principle** (agent-native memory study, arXiv 2606.24775) already establishes that write-time extraction should preserve context. DK-Mem does not claim this insight; it tests whether preservation alone is enough for merge decisions.
- **A-MAC**, **SAMD**, the **Stanford sieve**, and **deterministic freshness aggregation** establish that cheap deterministic rules can beat LLM judgment. DK-Mem's results are presented as consistent with this, not as a new finding.

---

## 9. Explicitly dropped and invalid claims

The paper does **not** claim any of the following:

1. "Memory systems normalize multilingual input to English before storing it." Mem0's default prompt and Hindsight both keep the user's language. Where the distinction is lost is measured, not assumed.
2. "Merge gating is a new memory primitive" or "`compatible()` is a new primitive." It is attribute agreement and a cannot-link constraint.
3. "Write-time gating" as a contribution. MOSAIC, Verification-Gated Persona State Transitions, and A-MAC already gate writes.
4. "Deterministic rules beating LLM judgment" as a new finding.
5. "We invert cross-lingual entity alignment." Entity resolution already decides whether to merge. The point is that the evidence was destroyed upstream by the system's own processing.
6. "Language commitment as deferred binding" or the database-normalization analogy as a contribution. This is covered by the Late Filtering Principle and appears only as one sentence of motivation.
7. "No prior multilingual memory work exists" or "multilingual work only measures answer degradation." Multilingual memory retrieval has been benchmarked (e.g., Wontopos Tablet 2).
8. "Existing benchmarks have no write stage." HaluMem and MOSAIC have one.
9. "Nobody detects false identity." Coreference resolution and DPCA's source-monitoring errors both address forms of it. The claim is limited to *language-induced* false identity in agent memory.
10. The FCR/MCR pair or the τ-swept frontier as methodological contributions. They are required rigour.
11. The *kal* (yesterday/tomorrow) example as an instance of this mechanism. It is a tense mis-resolution, not two distinct values collapsing into one gloss, and is out of scope.
12. LightMem as a baseline, and the arXiv ID 2604.07798 for it. LightMem is arXiv 2510.18866, and one independent study classifies its long-term store as append-only, so it may never merge.
13. A definitive cross-lingual novelty claim. The wording is "to our knowledge, no prior work studies consolidation errors in agent memory under cross-lingual or code-mixed input" until the remaining scoop searches are complete.

---

## 10. Relationship to the existing DK-Mem implementation

**Kept as is**

- The Mem0 and A-Mem evaluation harness, using their real extraction and update prompts
- Distinction tags as metadata on memory entries
- The three-way compatibility gate (compatible / incompatible / underdetermined) at merge time
- No model training
- Kinbank as the source of the distinction-class lexicon
- LinCE for realistic Hinglish phrasing in probe sentences
- LoCoMo as the multi-session JSON scaffold
- FCR as the central metric
- The Team A / Team B split and the G1–G4 gate structure

**Reframed (no code change required)**

- The gate is described as a cannot-link constraint adapted from coreference resolution, not as a new primitive.
- The motivating premise changes from "systems translate to English" to "the distinction can be lost at four stages, and we measure which."

**Added**

- Config B: Mem0 run with its **default native-language prompt**, with the extractor's output language logged
- Config C: **verbatim surface storage** with the host merge judge
- Config D: **embedding-threshold merging** (bge-m3)
- **Raise-τ** and **prompt-informed merge judge** baselines
- **MCR** and the **τ sweep**, so results are reported as a frontier
- **Same-entity control pairs** in the probe set
- **Bloat measurement** (entries per conversation, unresolved links)
- A **non-Indic Kinbank slice** (~30 items)

**Removed from the core**

- Read-side distinction-aware disambiguation (answering differently based on the query's distinction key). It is not needed to establish the core claim and is cut unless time remains.
- LightMem as a baseline.
- Evidentiality, classifier, and deixis distinction classes.

---

## 11. Final scope: what we will actually build

**In scope**

1. **Probe set.** ~200 Tier 1 minimal pairs (kinship primary; honorific/register and name variants secondary; same-entity controls in every class), ~60 Tier 2 injected LoCoMo conversations, ~30 Tier 3 non-Indic kinship items. Gold labels with κ.
2. **Distinction lexicon.** Kinship slice generated from Kinbank, plus honorific and name-variant entries. Coverage reported.
3. **Four-configuration pipeline harness** (A–D) on Mem0 and A-Mem, with Qwen2.5-1.5B / 3B / 7B backbones and output-language logging.
4. **DK-Mem gate** in lexicon-only and lexicon + LLM variants, insertable in front of both LLM-judge and embedding-threshold merging.
5. **Baselines** as listed in Section 6.3.
6. **Evaluation:** FCR, MCR, τ-swept frontier, stage-attributed FCR, downstream QA on destroyed facts, bloat, paired bootstrap CIs over ≥3 seeds.
7. **Ecological frequency estimate** of in-scope distinction classes in LinCE / IndicTalk.
8. **Release:** probe set, lexicon, and reproducibility package.

**Build order (30 days)**

| Week | Work | Decision point |
|---|---|---|
| 1 | Harness for configs A–D on Mem0 / A-Mem; 30-pair pilot; lexicon v1 from Kinbank; remaining scoop searches | Is FCR substantial (H1)? Does it persist in configs B–D (H2)? If not, narrow to the English-forced setting |
| 2 | Tier 1 complete with κ; all baselines run, including store-surface-only and raise-τ; first FCR–MCR numbers | Does DK-Mem beat raise-τ at matched MCR on Tier 1? |
| 3 | Tier 2 injection and runs; Tier 3 slice; backbone sweep; ablations; bloat; ecological estimate | Scope frozen |
| 4 | Writing; figures (FCR–MCR frontier, stage-attribution chart, *chachi/mausi* worked example, per-class breakdown); release package | Submission |

**Out of scope**

- Model training or fine-tuning
- Read-side disambiguation
- New memory architectures beyond the gate
- More than one non-Indic language
- Evidentiality, classifier, and temporal-deixis (*kal*) cases
- Claims about systems not tested (e.g., Hindsight is cited as a motivating example only)

---

## 12. One-sentence pitch

> Agent memory systems merge facts that look the same, and for multilingual users two facts that are different in Hindi can look the same to the system, so it silently deletes one; we measure where in the memory pipeline the difference is lost, test whether keeping the original language is enough, and show that a near-zero-cost dictionary-based cannot-link rule blocks those merges better than simply being more cautious.
