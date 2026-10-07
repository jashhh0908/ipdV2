"""Build the Kaggle notebook that runs the candidate Config B prompt on all 173 Tier 1
records with Qwen2.5-1.5B, 3B and 7B-Instruct (same decoding for all).

Per model: fp16, greedy, seed 0, batch 8, max_new_tokens 256 (the settings of the
A-C smoke test). 1.5B and 3B load on cuda:0 as before; 7B in fp16 does not fit one
T4, so it is sharded over both T4s with device_map (same dtype and decoding).
No merge judge, no Config A/C/D, no baselines. The candidate prompt is the unmodified
validation/native_b_candidate/candidate_prompts.py (sha256 6ca296ec...).

Usage: python build_kernel.py <kaggle-username> <out-dir>
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
USER, OUT = sys.argv[1], os.path.abspath(sys.argv[2])
SLUG = "dk-mem-native-b-prefreeze"
TIER1 = "dkmem/data/tier1/eval_input/teamA_tier1_v2.jsonl"
TIER1_SHA256 = "eab5dcb23375eb2e744c72a98cb31c25da0171b2ba3297a0843574ed9a34cd07"
CANDIDATE_SHA256 = "6ca296ec04dc04c02bb7911618359f6ea82aaab25c92edec74d3bcfe635716af"
MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]

FILES = ["dkmem/__init__.py", "dkmem/backends/__init__.py", "dkmem/backends/llm.py",
         "dkmem/memory/__init__.py", "dkmem/memory/prompts.py"]
embedded = {}
for rel in FILES:
    with open(os.path.join(REPO, rel), encoding="utf-8", newline="") as f:
        embedded[rel] = f.read()
with open(os.path.join(REPO, "validation", "native_b_candidate", "candidate_prompts.py"), encoding="utf-8", newline="") as f:
    embedded["candidate_prompts.py"] = f.read()
raw = open(os.path.join(REPO, TIER1), "rb").read()
assert hashlib.sha256(raw).hexdigest() == TIER1_SHA256, "Tier 1 input changed"
tier1_text = raw.decode("utf-8")


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md("# Candidate Config B prompt on all 173 Tier 1 records: Qwen2.5 1.5B / 3B / 7B"),
    code(f"""
import gc, hashlib, json, os, subprocess, sys, time
GPU_LIST = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print(GPU_LIST)
SRC = "/kaggle/working/src"
FILES = json.loads({json.dumps(json.dumps(embedded, ensure_ascii=False))})
for rel, text in FILES.items():
    path = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
TIER1_TEXT = {json.dumps(tier1_text, ensure_ascii=False)}
assert hashlib.sha256(TIER1_TEXT.encode("utf-8")).hexdigest() == "{TIER1_SHA256}"
sys.path.insert(0, SRC)
FILE_SHA256 = {{r: hashlib.sha256(t.encode("utf-8")).hexdigest() for r, t in FILES.items()}}
"""),
    code(f"""
import torch
from dkmem.backends.llm import HFGenerator, ModelConfig, GenerationParams, _dtype_kwarg
import candidate_prompts as CP
assert CP.NATIVE_B_CANDIDATE.sha256 == "{CANDIDATE_SHA256}", CP.NATIVE_B_CANDIDATE.sha256
print("candidate prompt", CP.NATIVE_B_CANDIDATE.prompt_id, CP.NATIVE_B_CANDIDATE.sha256)
PARAMS = GenerationParams(max_new_tokens=256, seed=0, batch_size=8)  # greedy, as in the smoke tests
RECORDS = [json.loads(l) for l in TIER1_TEXT.splitlines() if l.strip()]
SIDES = [(r, s) for r in RECORDS for s in ("a", "b")]
UTTS = [r["utterance_" + s] for r, s in SIDES]
print(len(RECORDS), "records,", len(UTTS), "sides; visible GPUs:", torch.cuda.device_count())

def load_generator(model_id, shard):
    cfg = ModelConfig(model_id=model_id)
    if not shard:
        return HFGenerator.load(cfg)
    # same loading recipe as HFGenerator.load, but fp16 weights split over both T4s
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import GenerationConfig as HFGenerationConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", max_memory={{0: "10GiB", 1: "12GiB"}}, **_dtype_kwarg(torch.float16))
    model.eval()
    shipped = model.generation_config
    model.generation_config = HFGenerationConfig(
        bos_token_id=shipped.bos_token_id, eos_token_id=shipped.eos_token_id, pad_token_id=tok.pad_token_id)
    return HFGenerator(model, tok, cfg, device_name="2x Tesla T4 (sharded)")
"""),
    code(f"""
RESULTS, ERRORS = {{}}, {{}}
def run_model(model_id):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    gen = load_generator(model_id, shard=model_id.endswith("7B-Instruct"))
    load_seconds = round(time.time() - t0, 1)
    info = gen.run_info(PARAMS)
    t1 = time.time()
    outs = gen.generate([CP.NATIVE_B_CANDIDATE.render(u) for u in UTTS], PARAMS)
    gen_seconds = round(time.time() - t1, 1)
    sides = [{{"eval_pair_id": r["eval_pair_id"], "side": s, "language": r["language"],
              "utterance": u, "raw_output": o}} for (r, s), u, o in zip(SIDES, UTTS, outs)]
    peak = [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())]
    RESULTS[model_id] = {{"run_info": info, "sides": sides, "load_seconds": load_seconds,
                         "generate_seconds": gen_seconds, "peak_gpu_memory_gb": peak}}
    name = model_id.split("/")[-1]
    with open(f"/kaggle/working/native_b_{{name}}.json", "w", encoding="utf-8") as f:
        json.dump({{"model_id": model_id, "prompt_id": CP.NATIVE_B_CANDIDATE.prompt_id,
                   "prompt_sha256": CP.NATIVE_B_CANDIDATE.sha256, "params": PARAMS.__dict__,
                   "tier1_sha256": "{TIER1_SHA256}", "nvidia_smi_L": GPU_LIST,
                   "file_sha256": FILE_SHA256, **RESULTS[model_id]}}, f, indent=1, ensure_ascii=False)
    print(model_id, "load", load_seconds, "s | generate", gen_seconds, "s | peak GB", peak, "|",
          info["resolved_revision"], flush=True)
    del gen

import traceback
for model_id in {json.dumps(MODELS)}:
    try:
        run_model(model_id)
    except Exception:
        ERRORS[model_id] = traceback.format_exc()
        print("FAILED", model_id, ERRORS[model_id], flush=True)
    gc.collect()
    torch.cuda.empty_cache()
with open("/kaggle/working/errors.json", "w", encoding="utf-8") as f:
    json.dump(ERRORS, f, indent=1)
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(OUT, exist_ok=True)
nb_name = "native_b_prefreeze.ipynb"
with open(os.path.join(OUT, nb_name), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
meta = {"id": f"{USER}/{SLUG}", "title": "DK-Mem Native B Prefreeze", "code_file": nb_name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}
with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("wrote", OUT, os.path.getsize(os.path.join(OUT, nb_name)), "bytes")
