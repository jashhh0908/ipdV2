"""Build the Kaggle notebook that compares the candidate Config B prompt with the
current one (Mem0 legacy FACT_RETRIEVAL_PROMPT) on the same 20 Tier 1 records.

Same model (Qwen2.5-3B-Instruct, ModelConfig defaults), decoding (greedy, seed 0,
batch 8, max_new_tokens 256), merge judge (merge_judge_v1, max_new_tokens 128),
gate (apply_gate, lexicon mode) and language report as the A-C smoke test.
Nothing in dkmem/ is modified; the candidate prompt lives in candidate_prompts.py.

Usage: python build_kernel.py <kaggle-username> <out-dir>
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
USER, OUT = sys.argv[1], os.path.abspath(sys.argv[2])
SLUG = "dk-mem-native-b-candidate"
SLICE_PATH = os.path.join(REPO, "validation", "smoke_abc", "kaggle_output", "tier1_slice.jsonl")
SLICE_SHA256 = "fb2fd613246b0cfb410e7f9445e5993f00e482ef8ef66895bcea230d6dd06d84"

FILES = [
    "dkmem/__init__.py",
    "dkmem/config.py",
    "dkmem/backends/__init__.py",
    "dkmem/backends/llm.py",
    "dkmem/memory/__init__.py",
    "dkmem/memory/cache.py",
    "dkmem/memory/consolidation.py",
    "dkmem/memory/dkmem_extract.py",
    "dkmem/memory/distinction_features.json",
    "dkmem/memory/extract.py",
    "dkmem/memory/gate.py",
    "dkmem/memory/judge.py",
    "dkmem/memory/lexicon.py",
    "dkmem/memory/matcher.py",
    "dkmem/memory/native_extract.py",
    "dkmem/memory/prompts.py",
    "dkmem/memory/schema.py",
    "dkmem/memory/scope.py",
    "dkmem/memory/similarity.py",
    "dkmem/pipeline/__init__.py",
    "dkmem/pipeline/trace.py",
]
embedded = {}
for rel in FILES:
    with open(os.path.join(REPO, rel), encoding="utf-8", newline="") as f:
        embedded[rel] = f.read()
with open(os.path.join(HERE, "candidate_prompts.py"), encoding="utf-8", newline="") as f:
    embedded["candidate_prompts.py"] = f.read()
with open(SLICE_PATH, encoding="utf-8", newline="") as f:
    slice_text = f.read()
assert hashlib.sha256(slice_text.encode("utf-8")).hexdigest() == SLICE_SHA256, "slice changed"


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md("# Config B: candidate native-language prompt vs current (Mem0 legacy) prompt, 20 Tier 1 records"),
    code(f"""
import hashlib, json, os, subprocess, sys, time
from dataclasses import replace
GPU_LIST = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print(GPU_LIST)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SRC = "/kaggle/working/src"
FILES = json.loads({json.dumps(json.dumps(embedded, ensure_ascii=False))})
for rel, text in FILES.items():
    path = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
SLICE_TEXT = {json.dumps(slice_text, ensure_ascii=False)}
assert hashlib.sha256(SLICE_TEXT.encode("utf-8")).hexdigest() == "{SLICE_SHA256}"
sys.path.insert(0, SRC)
FILE_SHA256 = {{r: hashlib.sha256(t.encode("utf-8")).hexdigest() for r, t in FILES.items()}}
"""),
    code("""
import torch
from dkmem.backends.llm import HFGenerator, ModelConfig, GenerationParams
from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.native_extract import MEM0_DEFAULT_FACTS_V1, native_extract_many, output_language_report
from dkmem.memory.judge import MERGE_JUDGE_V1, judge_pairs
from dkmem.memory.gate import apply_gate
from dkmem.pipeline.trace import diagnose, lexicon_distinction, pair_input_hash
from dkmem.memory.prompts import MEM0_EXTRACTION_V4
"""),
    code("""
import candidate_prompts as CP
assert MEM0_DEFAULT_FACTS_V1.sha256 == "e9c4726fffc75febe5e5090b0d38305c76079165d54a9ee38e265c02877ad469"
assert MEM0_EXTRACTION_V4.sha256 == "3b4e1de73ac89d5142d0e0cfba52a18e03307f87ac69b2646fb1c9abbabdf00a"
print("candidate prompt", CP.NATIVE_B_CANDIDATE.prompt_id, CP.NATIVE_B_CANDIDATE.sha256)
gen = HFGenerator.load(ModelConfig())
PARAMS = GenerationParams(max_new_tokens=256, seed=0, batch_size=8)
JUDGE_PARAMS = replace(PARAMS, max_new_tokens=128)
info = gen.run_info(PARAMS)
print(json.dumps(info, indent=2))
LEX = load_lexicon(os.path.join(SRC, "dkmem/memory/distinction_features.json"))
RECORDS = [json.loads(l) for l in SLICE_TEXT.splitlines() if l.strip()]
SIDES = [(r, s) for r in RECORDS for s in ("a", "b")]
UTTS = [r["utterance_" + s] for r, s in SIDES]
print(len(RECORDS), "records,", len(UTTS), "sides")
"""),
    code("""
def parse_candidate(raw):
    # strict: exactly {"fact": non-empty string}; no fence stripping, no repair
    try:
        d = json.loads(raw)
    except Exception as e:
        return None, f"not a single JSON object ({e})"
    if not isinstance(d, dict) or set(d) != {"fact"}:
        return None, f"keys must be exactly ['fact'], got {d!r}"
    if not isinstance(d["fact"], str) or not d["fact"].strip():
        return None, "fact must be a non-empty string"
    return d["fact"], None

def extract_candidate():
    outs = gen.generate([CP.NATIVE_B_CANDIDATE.render(u) for u in UTTS], PARAMS)
    sides = []
    for (r, s), utt, raw in zip(SIDES, UTTS, outs):
        fact, err = parse_candidate(raw)
        sides.append({"eval_pair_id": r["eval_pair_id"], "side": s, "language": r["language"], "utterance": utt,
                      "raw_output": raw, "facts": [fact] if fact else [], "error": err})
    return sides

def extract_current():
    res = native_extract_many(UTTS, gen, PARAMS)
    sides = []
    for (r, s), utt, e in zip(SIDES, UTTS, res):
        sides.append({"eval_pair_id": r["eval_pair_id"], "side": s, "language": r["language"], "utterance": utt,
                      "raw_output": e.raw_output, "facts": list(e.facts or []), "error": e.error,
                      "fence_stripped": e.fence_stripped})
    return sides

def downstream(sides):
    # unchanged judge + gate + diagnosis on every a x b fact pair
    by = {(x["eval_pair_id"], x["side"]): x for x in sides}
    cands = []
    for r in RECORDS:
        a, b = by[(r["eval_pair_id"], "a")], by[(r["eval_pair_id"], "b")]
        for i, ta in enumerate(a["facts"]):
            for j, tb in enumerate(b["facts"]):
                cands.append((r, i, j, ta, tb))
    judged = judge_pairs([(c[3], c[4]) for c in cands], gen, JUDGE_PARAMS) if cands else []
    rows = []
    for (r, i, j, ta, tb), jr in zip(cands, judged):
        lang = r["language"]
        da, db = lexicon_distinction(LEX, r["utterance_a"], lang), lexicon_distinction(LEX, r["utterance_b"], lang)
        row = {"eval_pair_id": r["eval_pair_id"], "entry_a": ta, "entry_b": tb, "judge_raw": jr.raw_output,
               "judge_decision": jr.decision, "judge_reason": jr.reason, "judge_error": jr.error}
        if jr.decision:
            g = apply_gate(jr.decision, da, db)
            dg = diagnose(lexicon=LEX, lang=lang, utterance_a=r["utterance_a"], utterance_b=r["utterance_b"],
                          stored_text_a=ta, stored_text_b=tb, host_proposed=jr.decision,
                          storage_loss_stage="L2", host_loss_stage="L3")
            row.update(gate_compatibility=g.compatibility, gate_decision=g.decision, vetoed=g.vetoed,
                       first_loss_point=dg["first_loss_point"], visible_a=dg["visible_in_stored_a"],
                       visible_b=dg["visible_in_stored_b"], lexicon_a=da, lexicon_b=db)
        rows.append(row)
    return rows

OUT = {}
for name, fn in (("current_B_mem0_legacy", extract_current), ("candidate_B_native", extract_candidate)):
    t0 = time.time()
    sides = fn()
    for x in sides:
        x["language_reports"] = [output_language_report(x["utterance"], f, x["language"]) for f in x["facts"]]
    rows = downstream(sides)
    OUT[name] = {"sides": sides, "pairs": rows, "seconds": round(time.time() - t0, 1)}
    print(name, "sides parsed:", sum(1 for x in sides if x["error"] is None), "/", len(sides),
          "| with a fact:", sum(1 for x in sides if x["facts"]), "| pairs judged:", len(rows),
          "| seconds:", OUT[name]["seconds"])

with open("/kaggle/working/native_b_comparison.json", "w", encoding="utf-8") as f:
    json.dump({"outputs": OUT, "run_info": info, "params": PARAMS.__dict__, "judge_params": JUDGE_PARAMS.__dict__,
               "prompts": {"current": MEM0_DEFAULT_FACTS_V1.sha256, "candidate": CP.NATIVE_B_CANDIDATE.sha256,
                           "judge": MERGE_JUDGE_V1.sha256, "v4": MEM0_EXTRACTION_V4.sha256},
               "candidate_prompt_system": CP.SYSTEM, "nvidia_smi_L": GPU_LIST, "file_sha256": FILE_SHA256,
               "slice_sha256": "SLICE_SHA256_PLACEHOLDER", "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
              f, indent=2, ensure_ascii=False)
""".replace("SLICE_SHA256_PLACEHOLDER", SLICE_SHA256)),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(OUT, exist_ok=True)
nb_name = "native_b_candidate.ipynb"
with open(os.path.join(OUT, nb_name), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
meta = {"id": f"{USER}/{SLUG}", "title": "DK-Mem Native B Candidate", "code_file": nb_name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}
with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("wrote", OUT, os.path.getsize(os.path.join(OUT, nb_name)), "bytes")
