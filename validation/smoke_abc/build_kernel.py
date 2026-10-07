"""Build the self-contained Kaggle notebook for the Config A-C real-model smoke test.

Runs Configs A, B and C of the stage-attribution harness (dkmem.pipeline) over
the first 20 records of the sanctioned Tier 1 input, in all three DK-Mem modes,
with Qwen2.5-3B-Instruct (ModelConfig defaults: fp16, cuda:0, greedy, seed 0,
batch size 8, max_new_tokens 256). Config A is run twice to check that the
trace is reproducible. No Config D, no bge-m3, no baselines.

Usage: python build_kernel.py <kaggle-username> <out-dir>
Embeds the dkmem source, a few unit-test files and the 20-record slice.
"""

import hashlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USER, OUT = sys.argv[1], os.path.abspath(sys.argv[2])
SLUG = "dk-mem-abc-smoke-test"
LIMIT = 20
TIER1_INPUT = "dkmem/data/tier1/eval_input/teamA_tier1_v2.jsonl"
TIER1_SHA256 = "eab5dcb23375eb2e744c72a98cb31c25da0171b2ba3297a0843574ed9a34cd07"

FILES = [
    "dkmem/__init__.py",
    "dkmem/config.py",
    "dkmem/pairwise_eval.schema.json",
    "dkmem/run_manifest.schema.json",
    "dkmem/backends/__init__.py",
    "dkmem/backends/llm.py",
    "dkmem/backends/embedding.py",
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
    "dkmem/pipeline/runner.py",
    "dkmem/pipeline/trace.py",
    "dkmem/pipeline/cli.py",
    "dkmem/tier1/__init__.py",
    "dkmem/tier1/extraction.py",
    "dkmem/tier1/io.py",
    "dkmem/tier1/runner.py",
    "tests/test_pipeline.py",
    "tests/test_prompts.py",
    "tests/test_extract.py",
    "tests/test_gate.py",
    "tests/test_config.py",
    "tests/test_schema.py",
    "tests/test_similarity.py",
    "tests/test_consolidation.py",
    "tests/fixtures/synthetic_fixtures.py",
    "tests/fixtures/probe_items.jsonl",
    "tests/fixtures/extractions.jsonl",
    "tests/fixtures/merge_events.jsonl",
]
embedded = {}
for rel in FILES:
    with open(os.path.join(REPO, rel), encoding="utf-8", newline="") as f:
        embedded[rel] = f.read()

with open(os.path.join(REPO, TIER1_INPUT), "rb") as f:
    raw = f.read()
assert hashlib.sha256(raw).hexdigest() == TIER1_SHA256, "Tier 1 input hash changed"
slice_lines = raw.decode("utf-8").splitlines()[:LIMIT]
SLICE = "\n".join(slice_lines) + "\n"


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# DK-Mem Configs A-C: real-model smoke test (Qwen2.5-3B-Instruct, 20 Tier 1 records)
Outputs in `/kaggle/working/`: `runs/<group_id>/...` per config, `run_environment.json`, `determinism.json`.
"""),
    code(f"""
import hashlib, json, os, subprocess, sys, time
GPU_LIST = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print(GPU_LIST)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # decode on one T4, as in the earlier runs
SRC = "/kaggle/working/dkmem-src"
FILES = json.loads({json.dumps(json.dumps(embedded, ensure_ascii=False))})
for rel, text in FILES.items():
    path = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
SLICE_TEXT = {json.dumps(SLICE, ensure_ascii=False)}
SLICE_PATH = os.path.join(SRC, "tier1_slice.jsonl")
with open(SLICE_PATH, "w", encoding="utf-8", newline="") as f:
    f.write(SLICE_TEXT)
sys.path.insert(0, SRC)
FILE_SHA256 = {{r: hashlib.sha256(t.encode("utf-8")).hexdigest() for r, t in FILES.items()}}
SLICE_SHA256 = hashlib.sha256(SLICE_TEXT.encode("utf-8")).hexdigest()
print("slice sha256", SLICE_SHA256)
try:
    import jsonschema
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jsonschema"], check=True)
    import jsonschema
"""),
    code("""
p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                   cwd=SRC, capture_output=True, text=True)
UNIT_TESTS = p.stderr.strip().splitlines()[-1]
print(p.stderr[-2500:])
"""),
    code(f"""
import torch
from dkmem.backends.llm import HFGenerator, ModelConfig, GenerationParams
from dkmem.memory.lexicon import load_lexicon
from dkmem.pipeline.runner import build_run_config, run_stage_attribution, write_outputs
from dkmem.tier1.io import Tier1Record

print("visible CUDA devices after restriction:", torch.cuda.device_count())
gen = HFGenerator.load(ModelConfig())
PARAMS = GenerationParams(max_new_tokens=256, seed=0, batch_size=8)  # greedy
info = gen.run_info(PARAMS)
print(json.dumps(info, indent=2))
LEX_PATH = os.path.join(SRC, "dkmem/memory/distinction_features.json")
LEX = load_lexicon(LEX_PATH)
RECORDS = [Tier1Record.from_dict(json.loads(l)) for l in SLICE_TEXT.splitlines() if l.strip()]
INPUT_INFO = {{"source_path": "first {LIMIT} lines of {TIER1_INPUT}", "file_sha256": SLICE_SHA256,
               "sanctioned": False, "parent_file_sha256": "{TIER1_SHA256}", "limit": {LIMIT}}}
MODES = ["off", "lexicon", "lexicon+llm"]
print(len(RECORDS), "records")
with open("/kaggle/working/run_environment.json", "w", encoding="utf-8") as f:
    json.dump({{"nvidia_smi_L": GPU_LIST, "run_info": info, "params": PARAMS.__dict__,
               "unit_tests": UNIT_TESTS, "file_sha256": FILE_SHA256, "slice_sha256": SLICE_SHA256}},
              f, indent=2, ensure_ascii=False)
"""),
    code("""
class CountingGenerator:
    def __init__(self, inner):
        self.inner, self.prompts = inner, 0
    @property
    def backbone(self):
        return self.inner.backbone
    def run_info(self, params=None):
        return self.inner.run_info(params)
    def generate(self, prompts, params=None):
        self.prompts += len(prompts)
        return self.inner.generate(prompts, params)

def run_config(config_id, out_root="/kaggle/working/runs"):
    g = CountingGenerator(gen)
    cfg = build_run_config(config_id, MODES, RECORDS, generator=g, lexicon_path=LEX_PATH,
                           params=PARAMS, input_info=INPUT_INFO)
    t0 = time.time()
    res = run_stage_attribution(RECORDS, config_id, LEX, cfg, generator=g, params=PARAMS)
    seconds = round(time.time() - t0, 1)
    paths = write_outputs(os.path.join(out_root, cfg["group_id"]), res)
    print("=== config", config_id, cfg["group_id"], "| prompts generated:", g.prompts, "| seconds:", seconds)
    print(json.dumps(res.summary, ensure_ascii=False))
    return res, {"prompts_generated": g.prompts, "seconds": seconds, "paths": paths}

RESULTS, STATS = {}, {}
for c in ("A", "B", "C"):
    RESULTS[c], STATS[c] = run_config(c)
print("peak GPU memory (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))
"""),
    code("""
# Reproducibility: run Config A a second time and compare everything that should be deterministic.
res2, stats2 = run_config("A", out_root="/kaggle/working/runs_repeat")
a1, a2 = RESULTS["A"], res2
det = {
    "fingerprint_equal": a1.run_config["fingerprint"] == a2.run_config["fingerprint"],
    "group_id_equal": a1.run_config["group_id"] == a2.run_config["group_id"],
    "trace_equal": a1.trace == a2.trace,
    "pairwise_rows_equal": {m: [r.to_dict() for r in a1.rows_by_mode[m]] == [r.to_dict() for r in a2.rows_by_mode[m]]
                            for m in MODES},
}
print(json.dumps(det))
with open("/kaggle/working/determinism.json", "w", encoding="utf-8") as f:
    json.dump({"config_A_rerun": det, "stats": STATS, "rerun_stats": {k: v for k, v in stats2.items() if k != "paths"},
               "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}, f, indent=2)
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(OUT, exist_ok=True)
nb_name = "dkmem_abc_smoke_test.ipynb"
with open(os.path.join(OUT, nb_name), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
meta = {"id": f"{USER}/{SLUG}", "title": "DK-Mem ABC Smoke Test", "code_file": nb_name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}
with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("wrote", OUT, "| notebook bytes:", os.path.getsize(os.path.join(OUT, nb_name)),
      "| slice sha256", hashlib.sha256(SLICE.encode("utf-8")).hexdigest())
