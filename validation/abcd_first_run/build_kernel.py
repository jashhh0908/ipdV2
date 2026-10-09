"""Build the Kaggle notebook for a real A-D run: the whole test suite as a preflight, then
the exact `python -m dkmem.pipeline.cli ...` command (see DKMEM_FIRST_RUN.md).

The notebook embeds the `dkmem` package (code, schemas, lexicon, the sanctioned Tier 1
input) and the tests. It does not run anything by itself here; `kaggle kernels push`
saves the version and starts it.

Example (the first real run, Qwen2.5-3B, all 173 records, seed 0):

    python build_kernel.py --user jashnikumbhe --out kernel_3b --model-id Qwen/Qwen2.5-3B-Instruct \\
        --seed 0 --tau-d 0.80 0.85 0.90 --slug dk-mem-abcd-3b-s0

Add `--shard` for Qwen2.5-7B-Instruct (weights split over both T4s) and `--limit N` for a
smoke test. All arguments after the model are forwarded to the CLI as shown in the notebook.

Baselines (DKMEM_BASELINES.md): `--baseline mem0` or `--baseline flat` runs `python -m dkmem.baselines.cli` instead
(no `--tau-d`, `--configs` or `--modes`); `--judge-prompt merge_judge_informed_v1` with `--modes off` makes the A-D
command the prompt-informed-judge baseline. `--embedding-revision` pins bge-m3 for Config D / the baselines.
"""

import argparse
import hashlib
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--slug", required=True)
ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--revision", default=None, help="pin the LLM weights to this commit hash")
ap.add_argument("--shard", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tau-d", nargs="+", default=None)
ap.add_argument("--baseline", choices=("mem0", "flat"), default=None,
                help="run dkmem.baselines.cli instead of the A-D pipeline")
ap.add_argument("--judge-prompt", default=None, help="merge judge of Configs A-C (e.g. merge_judge_informed_v1)")
ap.add_argument("--embedding-revision", default=None)
ap.add_argument("--configs", nargs="+", default=["A", "B", "C", "D"])
ap.add_argument("--modes", nargs="+", default=["off", "lexicon", "lexicon+llm"])
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

if args.baseline is None and ("D" in args.configs) and not args.tau_d:
    ap.error("--tau-d is required when Config D is run")
if args.baseline is not None and (args.tau_d or args.judge_prompt):
    ap.error("--tau-d / --judge-prompt belong to the A-D pipeline, not to --baseline")

SKIP_DIRS = {"__pycache__"}
embedded = {}
for top in ("dkmem", "tests"):
    for dirpath, dirnames, filenames in os.walk(os.path.join(REPO, top)):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith((".py", ".json", ".jsonl", ".md", ".txt")):  # .txt: the vendored, hash-pinned Mem0 sources
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, REPO).replace(os.sep, "/")
                with open(path, "rb") as f:
                    embedded[rel] = f.read().replace(b"\r\n", b"\n").decode("utf-8")

if args.baseline is not None:
    CLI_MODULE = "dkmem.baselines.cli"
    cli_args = [args.baseline, "--seed", str(args.seed), "--out", "/kaggle/working/runs"]
    if args.baseline == "mem0":
        cli_args += ["--model-id", args.model_id]
        if args.revision:
            cli_args += ["--revision", args.revision]
        if args.shard:
            cli_args.append("--shard")
else:
    CLI_MODULE = "dkmem.pipeline.cli"
    cli_args = ["--configs", *args.configs, "--modes", *args.modes, "--model-id", args.model_id,
                "--seed", str(args.seed), "--out", "/kaggle/working/runs"]
    if "D" in args.configs:
        cli_args += ["--tau-d", *args.tau_d]
    if args.revision:
        cli_args += ["--revision", args.revision]
    if args.shard:
        cli_args.append("--shard")
    if args.judge_prompt:
        cli_args += ["--judge-prompt", args.judge_prompt]
if args.limit is not None:
    cli_args += ["--limit", str(args.limit)]
if args.embedding_revision:
    cli_args += ["--embedding-revision", args.embedding_revision]


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md(f"# DK-Mem {'baseline ' + args.baseline if args.baseline else 'stage attribution, Configs ' + ' '.join(args.configs)}: "
       f"{args.model_id}, seed {args.seed}"),
    code(f"""
import hashlib, json, os, subprocess, sys, time
GPU_LIST = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print(GPU_LIST)
SRC = "/kaggle/working/src"
FILES = json.loads({json.dumps(json.dumps(embedded, ensure_ascii=False))})
for rel, text in FILES.items():
    path = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
print(len(FILES), "files written")
try:
    import jsonschema
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jsonschema"], check=True)
"""),
    code("""
# preflight: the whole unit-test suite must pass on the Kaggle image before the run starts
p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=SRC,
                   capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC})
print(p.stderr[-1500:])
assert p.returncode == 0, "unit tests failed"
"""),
    code(f"""
ARGS = {json.dumps(cli_args)}
print("python -m {CLI_MODULE}", " ".join(ARGS), flush=True)
t0 = time.time()
proc = subprocess.Popen([sys.executable, "-m", "{CLI_MODULE}", *ARGS], cwd=SRC, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, env={{**os.environ, "PYTHONPATH": SRC}})
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print("exit code", rc, "| seconds", round(time.time() - t0, 1))
assert rc == 0
"""),
    code("""
import glob
for path in sorted(glob.glob("/kaggle/working/runs/*/summary.json")):
    s = json.load(open(path, encoding="utf-8"))
    print(os.path.basename(os.path.dirname(path)), json.dumps({k: s[k] for k in s if k not in ("note",)}, ensure_ascii=False)[:1200])
json.dump({"nvidia_smi_L": GPU_LIST, "cli_args": ARGS}, open("/kaggle/working/run_environment.json", "w"), indent=2)
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(args.out, exist_ok=True)
nb_name = "dkmem_abcd_run.ipynb"
with open(os.path.join(args.out, nb_name), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
meta = {"id": f"{args.user}/{args.slug}", "title": args.slug.replace("-", " ").title()[:60], "code_file": nb_name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}
with open(os.path.join(args.out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("wrote", args.out, "|", len(embedded), "files |", f"python -m {CLI_MODULE} " + " ".join(cli_args))
