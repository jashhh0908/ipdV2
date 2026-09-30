Cross-Lingual False Consolidation in Agent Memory
=================================================
 
### A method-first research plan (replaces the benchmark-first version of Idea 2)
 
**Working title:** _English Gloss Is Not Enough: Distinction-Keyed Consolidation for Multilingual Agent Memory_
 
**Constraints:** 2 months · undergrad team · Kaggle free GPU · ≤6 pages · IEEE workshop target.
 
1\. Why the original framing was too weak
-----------------------------------------
 
The benchmark version's claim was "memory systems degrade on non-English input." That claim is:
 
*   **Predictable.** Everyone expects it. Reviewers will say "of course."
    
*   **Already half-answered.** Cross-lingual RAG has a named solution taxonomy — tRAG (translate query), CrossRAG (translate documents to a pivot), MultiRAG (retrieve across languages) — with CrossRAG at EACL 2026 Findings. Language-preference bias in multilingual RAG is documented (Chirkova et al., 2024). Code-mixed IR has dedicated methods (ContrastiveMix, ConCSE, CMLFormer).
    
*   **Partly refuted in the obvious direction.** Cross-lingual BrowseComp-Plus (arXiv:2606.15345) found a _fully target-language_ environment is worse than English-prompt + non-English evidence (≈5–6 pp penalty), concluding the bottleneck is intrinsic multilingual reasoning capability, not a surface language-switching artifact. So "just keep everything native" is not a fix.
    
**A paper whose contribution is "we translated a benchmark and things got worse" will be desk-rejected or accepted as a poster and forgotten.** The measurement is still needed — but as an _ingredient_, not the contribution.
 
2\. The structural wedge: memory has a write stage, RAG does not
----------------------------------------------------------------
 
This is the entire basis of the idea. Hold onto it.
 
**Cross-lingual RAGAgent memory**Corpus originGiven, external, fixed**Constructed by the system from user utterances**StorageDocuments stored **verbatim**Facts **extracted, rewritten, normalized**Over timeStatic**Consolidated — entries merged, updated, superseded**Language commitmentMade at query time (reversible)**Made at write time (irreversible)**
 
Every cross-lingual RAG technique operates on the read side, because that is the only side RAG has. Memory has a write side, and the write side is where language commitment happens — _before the system knows what will be asked, and before it knows what it will later need to merge against._
 
This is the same structural error as caption-then-store in multimodal memory (MemLens: memory agents are "length-stable but lose visual fidelity under storage-time compression"), but with a crucial difference that makes it _more_ interesting rather than derivative: with images, deferring commitment is expensive. With text, keeping the original is nearly free — so the interesting failure is not information loss, it is **what the lost information was silently used for.**
 
It was used for merge decisions.
 
3\. The failure mode: cross-lingual false consolidation
-------------------------------------------------------
 
### Mechanism
 
1.  User utterance in language L contains a distinction D that English lexicalizes coarsely or not at all.
    
2.  Write-side extraction normalizes to English. D is destroyed.
    
3.  A later utterance produces a normalized fact that is **string-identical or embedding-adjacent** to the first, because D was the only thing separating them.
    
4.  The consolidation step merges them, or treats one as a supersession of the other.
    
5.  A true fact is deleted. **No read-side technique can recover it — it is not in the store.**
    
### Why this is not the same as "translation is lossy"
 
Lossy translation degrades _answer quality_ for the question about that fact. False consolidation **destroys an unrelated fact** and does so silently, with the system reporting high confidence. It converts a linguistic issue into a memory-integrity issue. That is the reframing that makes this a systems paper rather than an NLP-bias paper.
 
### Distinction classes to exercise (do not stop at kinship)
 
**ClassExampleEnglish collapse**Kinship specificity_chachi / mami / bua / mausi_all → "aunt"Honorific / register_tu / tum / aap_; Japanese _\-san/-sama_all → "you"Name-variant surface formPriya / प्रिया / Preeyamay or may not unifyEvidentiality / hearsayTurkish _\-mış_, Quechua evidentialsdropped entirelyClassifier / measureHindi/Chinese classifiersdroppedPoliteness-encoded relationshipKorean speech levelsdroppedSpatial/temporal deixis_yeh/woh_, _kal_ (= yesterday **and** tomorrow)ambiguous or dropped
 
**Include at least one non-Indic language** (German T-V, Japanese honorifics, or Turkish evidentiality). This costs you ~30 probe items and buys you the claim that the phenomenon is not an Indic quirk. Without it, a reviewer will say "this is a Hindi paper." With it, it's a general finding about normalization-based memory writes.
 
Note the _kal_ case specifically — Hindi uses one word for both "yesterday" and "tomorrow," disambiguated by verb tense. An extractor that normalizes to a date can get this backwards, which produces **temporal** false consolidation. That's an unusually vivid example for a paper figure.
 
4\. Method: Distinction-Keyed Memory (DK-Mem)
---------------------------------------------
 
### Design principle
 
> Separate the _language-committing_ operation from the _merge_ operation, and gate the second on evidence the first would have destroyed.
 
### Architecture (three components, all cheap)
 
**(a) Write: dual-field extraction**
 
Each memory entry stores:
 
Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   {    gloss:        "user's aunt lives in Pune",     # English, normalized — the retrieval key    distinction:  {kin: "chachi", script: "deva"},  # source-language discriminative features    surface:      "meri chachi Pune mein rehti hai", # verbatim original — free to keep    lang_profile: {hi: 0.7, en: 0.3, script_mix: 0.4}  }   `
 
The distinction field is extracted by a **lexicon-first, model-second** cascade:
 
1.  A curated distinction lexicon (kinship terms, honorifics, classifiers, evidential markers) — deterministic dictionary lookup, zero cost, high precision.
    
2.  A small LLM call **only** for spans the lexicon flags as ambiguous.
    
Report lexicon-only vs lexicon+LLM as an ablation. If lexicon-only captures most of the benefit, that is a _stronger_ result — a deterministic, zero-inference-cost fix.
 
**(b) Consolidate: merge gating**
 
The core contribution. Replace:
 
Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   merge(a, b)  if  sim(gloss_a, gloss_b) > τ   `
 
with:
 
Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   merge(a, b)  if  sim(gloss_a, gloss_b) > τ  AND  compatible(distinction_a, distinction_b)   `
 
compatible() is deterministic and has three outcomes:
 
*   **Compatible** — keys agree, or one is empty (unmarked) → merge permitted.
    
*   **Incompatible** — keys are distinct values of the same feature (_chachi_ vs _mausi_) → **merge blocked, both retained as separate entities.**
    
*   **Underdetermined** — one key present, one absent, and the feature is discriminative → merge blocked, entries linked with an unresolved flag rather than merged or duplicated.
    
That third branch matters and is where reviewers will look. Blocking every uncertain merge causes memory bloat; merging them causes corruption. Handling underdetermination explicitly — and measuring the bloat cost — is what separates this from a hack. This also connects to work on irreducible memory conflict (TANGLE, arXiv:2608.13921), which argues agents should recognize underdetermination rather than force a single answer. You are applying that principle at the _write_ stage instead of the answer stage, which is new.
 
**(c) Read: distinction-aware disambiguation**
 
If the query carries a distinction key ("meri _chachi_ kahan rehti hai?"), it selects among gloss-equivalent entries. If it doesn't, and multiple incompatible entries match, the agent surfaces the ambiguity rather than picking one.
 
### Cost profile
 
*   Lexicon lookup: free.
    
*   compatible(): a dictionary comparison. Free.
    
*   Extra storage: one short field per entry.
    
*   Extra LLM calls: zero in the lexicon-only variant.
    
**This is the pitch.** A near-zero-cost, deterministic write-side constraint that prevents a class of silent memory corruption. Cheap deterministic primitives beating LLM reasoning has precedent in this exact subfield (deterministic freshness aggregation, arXiv:2606.01435), which strengthens rather than weakens your framing.
 
5\. Novelty — stated precisely, with the boundary against prior work
--------------------------------------------------------------------
 
**N1. A named, measured failure mode: cross-lingual false consolidation.** Existing multilingual work measures _answer degradation_. This measures _memory-store corruption_ — entities wrongly merged, true facts deleted. Different dependent variable, different mechanism, and it is invisible to every existing multilingual benchmark because none of them have a write stage. _Boundary:_ cross-lingual entity alignment (multilingual KG literature) studies how to _correctly_ merge entities across language-specific KGs. It assumes a ground-truth alignment exists and seeks it. Here the question is inverted: **whether alignment should happen at all**, when the evidence that would forbid it was destroyed upstream. Cite the alignment literature and state this inversion explicitly — a reviewer familiar with that field will otherwise raise it.
 
**N2. Merge-gating as a memory primitive.** Consolidation in Mem0, A-Mem, LightMem, and Memory-R1 is a similarity-plus-LLM-judgment operation. Adding a _hard, deterministic, linguistically-motivated compatibility constraint_ — with an explicit underdetermined branch — is a new primitive, not a tuned threshold. _Boundary:_ MemRouter (arXiv:2605.00356) learns write-side _admission_ (store or not). DK-Mem governs write-side _identity_ (same entity or not). Orthogonal, and worth one sentence of related work.
 
**N3. Language commitment as a deferred-binding problem.** The transplant from adjacent fields: late binding in programming-language design, and lossless-vs-lossy normalization in database schema design. The general principle — _do not discard the discriminative field before performing the operation that needs it_ — is a normalization-theory argument (this is, structurally, a lossy decomposition that fails to preserve a functional dependency). Framing a multilingual NLP problem in database-normalization terms is a genuinely different lens and is the kind of framing that makes small papers memorable.
 
6\. Data
--------
 
You must **construct** the core test set. This is a feature, not a shortcut: the phenomenon is invisible in existing data by definition, since existing data is English.
 
**Tier 1 — Minimal-pair conflation probes (the core, ~200 items).** Pairs of facts that are _distinct in language L_ and _identical after English normalization_. Each item: two utterances, a gold label (same entity / different entity), and the distinction feature responsible. Hand-authored by your team, cross-validated by a second native speaker. 200 items is enough for a 6-page paper and is achievable in ~2 weeks of part-time team effort. **Do not chase scale here — chase clean labels and report κ.**
 
**Tier 2 — Injected multi-session conversations (~60 conversations).** Take LongMemEval-S / LoCoMo session skeletons, inject minimal-pair facts across distant sessions, and vary: language (Hindi / Hinglish / native-script / English control), session distance, and code-mixing ratio. This gives you a realistic setting and lets you measure false consolidation _in situ_ rather than in isolation.
 
**Tier 3 — Cross-lingual generalization probe (~30 items).** German T-V, Japanese honorifics, Turkish evidentiality. Small, but it converts "a Hindi finding" into "a normalization finding."
 
**Reference resources:** IndicTrans2 or NLLB-200-distilled-600M for translation passes; indic-transliteration for deterministic Romanization; XCR-Bench (arXiv:2601.14063) and MAKIEval for culture-specific-item taxonomy precedent; CVQA / ALM-Bench for cultural-benchmark construction conventions.
 
**Release the probe set.** A small, clean, human-validated diagnostic set with a novel failure mode is a legitimate secondary contribution.
 
7\. Baselines
-------------
 
**BaselineWhy it must be there**Mem0 (arXiv:2504.19413)Standard extract-and-consolidate; the direct targetA-Mem (arXiv:2502.12110)Linked-note consolidation, different merge logicLightMem (arXiv:2604.07798)SLM-native, matches your compute regimeFlat dense RAG, bge-m3No consolidation at all — **the honest controlStore-surface-onlyThe reviewer's first objection: "just don't normalize"**tRAG / CrossRAGThe cross-lingual RAG solutions, adapted to memoryDK-Mem (lexicon-only)Your deterministic variantDK-Mem (lexicon + LLM)Your full variant
 
**The store-surface-only baseline is the most important row in your table.** If it matches DK-Mem on everything, your paper collapses to "don't normalize," which is a much weaker claim. Your argument must be that it _loses on retrieval_ — which is plausible given the documented 30–50 point Hits@20 drop in cross-lingual retrieval and the finding that code-mixed representations are weakly anchored to both English and Hindi. DK-Mem's pitch is that it keeps the English retrieval key **and** the merge-safety. Run this baseline in week 3, not week 7.
 
8\. Evaluation
--------------
 
### Primary metric (new, and the paper's centre)
 
**False Consolidation Rate (FCR)** — the fraction of distinct-entity pairs that the system merges or treats as a supersession.
 
Report alongside its dual, to prevent the trivial degenerate solution of never merging:
 
**Missed Consolidation Rate (MCR)** — the fraction of same-entity pairs left unmerged.
 
**Report the FCR–MCR frontier**, swept over the similarity threshold τ. A single-point comparison is not credible; a system that never merges has FCR = 0 and is useless. The claim to establish is that **DK-Mem dominates the baselines' frontier** — strictly lower FCR at matched MCR. This is the figure the paper lives or dies on.
 
### Secondary
 
*   **Downstream QA accuracy** on questions whose answers were destroyed by false consolidation. This is what converts a store-integrity metric into something a reviewer cares about.
    
*   **Memory bloat**: entries retained per conversation. Quantifies the cost of the underdetermined branch.
    
*   **Ablations**: lexicon-only vs +LLM; distinction class × FCR (which classes are most dangerous); extraction-prompt language (English prompt vs native prompt) — this last one is itself a publishable mini-result.
    
*   **Backbone sweep**: Qwen2.5-1.5B / 3B / 7B. Predict FCR rises as models shrink, which strengthens the deterministic-primitive argument.
    
### Statistics
 
Paired bootstrap over pairs, 95% CIs, ≥3 seeds. Report Cohen's κ on your team's gold labels. For a hand-constructed dataset this is not optional — it is the first thing a reviewer checks.
 
9\. Honest challenges — and the kill criteria
---------------------------------------------
 
**Run these checks in week 1–2. Do not proceed on faith.**
 
**C1. Does the failure actually occur?** Maybe Qwen, extracting from Hindi, naturally preserves "chachi." **Pilot this on day 3** with 30 hand-made examples across Mem0's actual extraction prompt. _Kill criterion:_ if FCR < 10% for baselines, the premise is too weak for a method paper. _But note:_ the failure is probably _induced by the prompt template_, since memory systems prompt in English and implicitly elicit English output. If so, "FCR as a function of extraction-prompt language" becomes a finding in its own right and the paper survives in modified form.
 
**C2. "You just kept the original text."** Answered only by the store-surface-only baseline losing on retrieval. If it doesn't lose, pivot: the paper becomes _"Normalization-based memory writes are merge-unsafe: an empirical case against write-time language commitment"_ — an empirical finding rather than a method. Weaker, still publishable at a workshop, but know this in week 3, not week 7.
 
**C3. Is the lexicon doing all the work, and is that cheating?** No — a deterministic lexicon achieving what an LLM cannot is the _interesting_ version of the result, consistent with deterministic freshness aggregation outperforming LLM judgment. But you must report lexicon coverage honestly: what fraction of real distinctions it catches, and performance on the uncovered remainder.
 
**C4. Two languages is thin.** Mitigated by Tier 3. Be explicit in limitations about what you have and haven't shown. A reviewer forgives a stated limitation and punishes a hidden one.
 
**C5. Constructed data can be accused of being contrived.** Mitigate with Tier 2 (injection into real benchmark conversation structures) and by reporting how often the distinction classes occur naturally — even a rough frequency estimate from a Hinglish corpus (LinCE, GLUECoS) supports ecological validity. Do this; it's a half-day of work and closes a serious line of attack.
 
**C6. Scoop risk.** Lower than most ideas in the original list — this sits in a gap between two literatures rather than inside a hot one. Still, check arXiv for "multilingual agent memory" and "cross-lingual memory consolidation" in week 1.
 
10\. Eight-week plan
--------------------
 
**WeekWorkGate**1Local backends for Mem0/A-Mem/LightMem. **C1 pilot (30 examples).** arXiv scoop check.**Go/no-go on premise**2Distinction lexicon v1. Tier 1 probe set (100 of 200 items). Annotation protocol + κ pilot.κ > 0.73Tier 1 complete. **Run all baselines incl. store-surface-only. First FCR numbers.Go/no-go on method**4DK-Mem implementation (both variants). Tier 2 injection pipeline.5Main experiments: FCR–MCR frontier, downstream QA, backbone sweep.6Ablations, Tier 3 cross-lingual probe, bloat measurement. **Freeze scope.**7Writing. Figures: the frontier plot, the _chachi/mausi_ worked example, distinction-class breakdown.8Revision, reproducibility package, probe-set release.
 
**Compute:** ~30–40 GPU-hours. Comfortably within legitimate Kaggle quota. The binding constraint is annotation time, which parallelizes across your team — the best available match between this project and a multi-person undergrad group.
 
11\. The one-sentence pitch
---------------------------
 
> Memory systems normalize multilingual user utterances to English before storing them, and then use those normalized forms to decide which memories refer to the same thing — so a distinction that English does not lexicalize is destroyed _before_ the merge decision that depends on it, silently conflating entities and deleting true facts. We name this cross-lingual false consolidation, measure it, and show a near-zero-cost deterministic merge-gating constraint largely eliminates it.
 
That is a mechanism, a measurement, and a fix, in one sentence, with a failure nobody has named. It is the strongest version of this idea I can construct that a four-person undergrad team can actually finish in two months.