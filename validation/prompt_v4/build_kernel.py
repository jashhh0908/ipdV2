"""Build the self-contained Kaggle notebook that runs extraction prompts v2, v3
and v4 on the same 20 inputs (the 10 synthetic probes x sides a/b in
tests/fixtures/probe_items.jsonl) with the same Qwen2.5-3B-Instruct setup.

Setup is identical to the earlier v1/v2/v3 validation: ModelConfig() defaults
(fp16, cuda:0), greedy decoding, seed 0, batch_size 8, max_new_tokens 256, items
ordered (probe, side a then b), one fresh cache per prompt.

Usage: python build_kernel.py <kaggle-username> <out-dir>
Embeds only source + synthetic fixtures (no Tier 1 data).
"""

import hashlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USER, OUT = sys.argv[1], os.path.abspath(sys.argv[2])
SLUG = "dk-mem-prompt-v4-validation"
PROMPT_IDS = ["mem0_extraction_v2", "mem0_extraction_v3", "mem0_extraction_v4"]

FILES = [
    "dkmem/__init__.py",
    "dkmem/backends/__init__.py",
    "dkmem/backends/llm.py",
    "dkmem/memory/__init__.py",
    "dkmem/memory/schema.py",
    "dkmem/memory/scope.py",
    "dkmem/memory/prompts.py",
    "dkmem/memory/extract.py",
    "dkmem/memory/cache.py",
    "tests/test_schema.py",
    "tests/test_llm.py",
    "tests/test_fixtures.py",
    "tests/test_extract.py",
    "tests/test_cache.py",
    "tests/test_prompts.py",
    "tests/fixtures/synthetic_fixtures.py",
    "tests/fixtures/probe_items.jsonl",
    "tests/fixtures/extractions.jsonl",
    "tests/fixtures/merge_events.jsonl",
]
embedded = {}
for rel in FILES:
    with open(os.path.join(REPO, rel), encoding="utf-8", newline="") as f:
        embedded[rel] = f.read()


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# DK-Mem extraction prompts v2 / v3 / v4 on the 20 synthetic probe sides (Qwen2.5-3B-Instruct)
Same inputs, same model and decoding for every prompt. Outputs in `/kaggle/working/`:
`extraction_validation_<prompt_id>.json`, `extraction_cache_<prompt_id>.jsonl`, `run_environment.json`.
"""),
    code(f"""
import hashlib, json, os, subprocess, sys, time
GPU_LIST = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print(GPU_LIST)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # decode on one T4, as in the earlier v1-v3 runs
SRC = "/kaggle/working/dkmem-src"
FILES = json.loads({json.dumps(json.dumps(embedded, ensure_ascii=False))})
for rel, text in FILES.items():
    path = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
sys.path.insert(0, SRC)
FILE_SHA256 = {{r: hashlib.sha256(t.encode("utf-8")).hexdigest() for r, t in FILES.items()}}
for r, h in FILE_SHA256.items():
    print(h[:12], r)
"""),
    code("""
p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                   cwd=SRC, capture_output=True, text=True)
UNIT_TESTS = p.stderr.strip().splitlines()[-1]
print(p.stderr[-2500:])
"""),
    code("""
import torch
from dkmem.backends.llm import HFGenerator, ModelConfig, GenerationParams
from dkmem.memory.schema import ProbeItem, read_jsonl
from dkmem.memory.prompts import get_prompt
from dkmem.memory.extract import ExtractionBatchError, ExtractionError
from dkmem.memory.cache import CacheKey, ExtractionCache, cached_extract_many

print("visible CUDA devices after restriction:", torch.cuda.device_count())
gen = HFGenerator.load(ModelConfig())
PARAMS = GenerationParams(max_new_tokens=256, seed=0, batch_size=8)  # greedy
info = gen.run_info(PARAMS)
BACKEND_INFO = {"revision": info["resolved_revision"], "dtype": info["model_config"]["dtype"]}
print(json.dumps(info, indent=2))
PROBES = list(read_jsonl(os.path.join(SRC, "tests/fixtures/probe_items.jsonl"), ProbeItem))
ITEMS = [(p, s) for p in PROBES for s in ("a", "b")]
print(len(PROBES), "probes,", len(ITEMS), "items")
with open("/kaggle/working/run_environment.json", "w", encoding="utf-8") as f:
    json.dump({"nvidia_smi_L": GPU_LIST, "run_info": info, "backend_info": BACKEND_INFO,
               "params": PARAMS.__dict__, "unit_tests": UNIT_TESTS, "file_sha256": FILE_SHA256},
              f, indent=2, ensure_ascii=False)
"""),
    code("""
class CountingGenerator:
    def __init__(self, inner):
        self.inner, self.prompts = inner, 0
    @property
    def backbone(self):
        return self.inner.backbone
    def generate(self, prompts, params=None):
        self.prompts += len(prompts)
        return self.inner.generate(prompts, params)

def failure_category(msg):
    if "not a single JSON object" in msg or "must be a JSON object" in msg or "duplicate JSON keys" in msg or "must be str" in msg:
        return "json"
    if "output keys" in msg:
        return "keys"
    if "'distinction" in msg:
        return "distinction_validation"
    if "'surface'" in msg:
        return "surface"
    if "'gloss'" in msg:
        return "gloss"
    return "other"

def diagnostic_parse(raw):
    # Diagnostics only: shows what the model attempted. Never used as a result.
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else None
    except Exception:
        return None

def run_prompt(prompt_id):
    PROMPT = get_prompt(prompt_id)
    print("=== prompt:", PROMPT.prompt_id, PROMPT.sha256)
    cache_path = f"/kaggle/working/extraction_cache_{prompt_id}.jsonl"
    if os.path.exists(cache_path):
        os.remove(cache_path)
    keys = [CacheKey.build(p, s, prompt=PROMPT, backbone=gen.backbone, params=PARAMS,
                           backend_info=BACKEND_INFO) for p, s in ITEMS]

    def run_pass(cache):
        counter = CountingGenerator(gen)
        hit = [k in cache for k in keys]
        failures = {}
        t0 = time.time()
        try:
            cached_extract_many(ITEMS, counter, cache, PARAMS, prompt=PROMPT, backend_info=BACKEND_INFO)
        except ExtractionBatchError as e:
            failures = {(f.pair_id, f.side): f for f in e.failures}
        return {"hit": hit, "failures": failures, "prompts_generated": counter.prompts,
                "seconds": round(time.time() - t0, 1)}

    cache = ExtractionCache(cache_path)
    pass1 = run_pass(cache)
    pass2 = run_pass(cache)
    reloaded = ExtractionCache(cache_path)  # full integrity verification from disk
    rows = []
    for i, (probe, side) in enumerate(ITEMS):
        utt = probe.utt_a if side == "a" else probe.utt_b
        value = probe.distinction_value_a if side == "a" else probe.distinction_value_b
        expected = {} if value is None else {probe.distinction_class.removeprefix("control_"): value}
        fail = pass1["failures"].get((probe.pair_id, side))
        ex = cache.get(keys[i])
        row = {"pair_id": probe.pair_id, "side": side, "lang": probe.lang,
               "distinction_class": probe.distinction_class, "utterance": utt,
               "expected_distinction": expected,
               "pass1_cache": "hit" if pass1["hit"][i] else "miss",
               "pass2_cache": "hit" if pass2["hit"][i] else "miss",
               "reloaded_from_disk": keys[i] in reloaded}
        if ex is not None:
            row.update(success=True, strict_validation=True, error=None, failure_category=None,
                       raw_output=ex.raw_output, gloss=ex.gloss, distinction=ex.distinction,
                       surface_exact_match=(ex.surface == utt), lang_profile=ex.lang_profile,
                       distinction_matches_expected=(ex.distinction == expected))
        else:
            d = diagnostic_parse(fail.raw_output)
            row.update(success=False, strict_validation=False, error=str(fail),
                       failure_category=failure_category(str(fail)), raw_output=fail.raw_output,
                       gloss=(d or {}).get("gloss"), distinction=(d or {}).get("distinction"),
                       surface_exact_match=(d.get("surface") == utt) if d and "surface" in d else None,
                       lang_profile=None, distinction_matches_expected=None)
        rows.append(row)
    bad = [r for r in rows if not r["success"]]
    cats = {}
    for r in bad:
        cats[r["failure_category"]] = cats.get(r["failure_category"], 0) + 1
    summary = {
        "prompt_id": PROMPT.prompt_id, "prompt_sha256": PROMPT.sha256,
        "items": len(rows), "successful": len(rows) - len(bad), "failed": len(bad),
        "failure_categories": cats,
        "cache": {"pass1_hits": sum(pass1["hit"]), "pass1_prompts_generated": pass1["prompts_generated"],
                  "pass2_hits": sum(pass2["hit"]), "pass2_prompts_generated": pass2["prompts_generated"],
                  "entries_on_disk_verified": len(reloaded)},
        "timing_seconds": {"pass1": pass1["seconds"], "pass2": pass2["seconds"]},
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    with open(f"/kaggle/working/extraction_validation_{prompt_id}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": rows, "run_info": info, "backend_info": BACKEND_INFO,
                   "params": PARAMS.__dict__, "prompt_id": PROMPT.prompt_id,
                   "prompt_sha256": PROMPT.sha256, "unit_tests": UNIT_TESTS,
                   "file_sha256": FILE_SHA256}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    for r in rows:
        print(f"{r['pair_id']}/{r['side']} ok={r['success']} dist={r['distinction']} exp={r['expected_distinction']} | {r['gloss']}")
    return summary

ALL = {}
for pid in PROMPT_IDS_PLACEHOLDER:
    ALL[pid] = run_prompt(pid)
print(json.dumps({k: {"ok": v["successful"], "failed": v["failed"]} for k, v in ALL.items()}))
"""),
]

for c in cells:
    c["source"] = [line.replace("PROMPT_IDS_PLACEHOLDER", json.dumps(PROMPT_IDS)) for line in c["source"]]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(OUT, exist_ok=True)
nb_name = "dkmem_prompt_v4_validation.ipynb"
with open(os.path.join(OUT, nb_name), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
meta = {"id": f"{USER}/{SLUG}", "title": "DK-Mem Prompt V4 Validation", "code_file": nb_name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}
with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
for rel, text in embedded.items():
    print(f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}  {len(text):6d} B  {rel}")
print("wrote", OUT, "| notebook bytes:", os.path.getsize(os.path.join(OUT, nb_name)))
