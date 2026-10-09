"""Build the Kaggle notebook for the limit=10 real-model smoke test of every system.

One notebook, Qwen2.5-3B-Instruct (revision pinned) + bge-m3 (revision pinned), the first 10 sanctioned Tier 1
records, seed 0, greedy:

  1. the whole unit-test suite (preflight);
  2. Configs A B C D x {off, lexicon, lexicon+llm}  (tau 0.85 for D), run twice (second run -> runs_repeat/);
  3. the prompt-informed judge (A B C, gate off);
  4. Mem0 OSS v1.0.11, run twice;
  5. never-merge / flat RAG, run twice;
  6. checks: schema validity of every output, identical fingerprints / traces / rows across the repeats, resolved
     model and embedder revisions, gating invariants, the offline evaluators on the real outputs (with a PLACEHOLDER
     gold key: wiring only, no FCR/MCR is reported).

Usage: python build_kernel.py --user <kaggle-user> --out <dir> [--slug dk-mem-baselines-smoke]
Then: kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4
"""

import argparse
import base64
import io
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--slug", default="dk-mem-baselines-smoke")
ap.add_argument("--limit", type=int, default=10)
ap.add_argument("--slice-incompatible", type=int, default=0,
                help="instead of the first N records, run a slice of N records: this many lexicon-incompatible pairs "
                     "(so the gate has something to veto) and the rest compatible, in input order. The slice is not "
                     "the sanctioned file (recorded as sanctioned: false).")
args = ap.parse_args()

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REV = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EMB_REV = "5617a9f61b028005a4858fdac845db406aefb181"

embedded = {}
for top in ("dkmem", "tests"):
    for dirpath, dirnames, filenames in os.walk(os.path.join(REPO, top)):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith((".py", ".json", ".jsonl", ".md", ".txt")):
                path = os.path.join(dirpath, fn)
                with open(path, "rb") as f:
                    embedded[os.path.relpath(path, REPO).replace(os.sep, "/")] = f.read().replace(b"\r\n", b"\n").decode("utf-8")

buf = io.BytesIO()  # the sources travel as a base64 zip: Kaggle rejects a ~1.5 MB plain-text notebook (HTTP 400)
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for rel, text in sorted(embedded.items()):
        z.writestr(zipfile.ZipInfo(rel, (2026, 10, 8, 0, 0, 0)), text.encode("utf-8"), zipfile.ZIP_DEFLATED)
PAYLOAD = base64.b64encode(buf.getvalue()).decode("ascii")

SLICED = args.slice_incompatible > 0
input_args = ["--input", "/kaggle/working/slice.jsonl"] if SLICED else ["--limit", str(args.limit)]
common = ["--model-id", MODEL, "--revision", MODEL_REV, "--embedding-revision", EMB_REV, "--seed", "0", *input_args]
COMMANDS = [
    ("abcd", "dkmem.pipeline.cli", ["--configs", "A", "B", "C", "D", "--modes", "off", "lexicon", "lexicon+llm", "--tau-d", "0.85", *common], "runs"),
    ("abcd_repeat", "dkmem.pipeline.cli", ["--configs", "A", "B", "C", "D", "--modes", "off", "lexicon", "lexicon+llm", "--tau-d", "0.85", *common], "runs_repeat"),
    ("informed", "dkmem.pipeline.cli", ["--configs", "A", "B", "C", "--modes", "off", "--judge-prompt", "merge_judge_informed_v1", *common], "runs"),
    ("mem0", "dkmem.baselines.cli", ["mem0", *common], "runs"),
    ("mem0_repeat", "dkmem.baselines.cli", ["mem0", *common], "runs_repeat"),
    ("flat", "dkmem.baselines.cli", ["flat", "--embedding-revision", EMB_REV, "--seed", "0", *input_args], "runs"),
    ("flat_repeat", "dkmem.baselines.cli", ["flat", "--embedding-revision", EMB_REV, "--seed", "0", *input_args], "runs_repeat"),
]

CHECKS = r'''
import glob, hashlib, itertools
from pathlib import Path
SLICED = SLICED_FLAG
sys.path.insert(0, SRC)
import jsonschema
from dkmem.eval.groups import load_group
from dkmem.eval.fairness import check_comparable
from dkmem.eval.gold import GoldPair, gold_from_pairs
from dkmem.eval.sweep import sweep_config_d
from dkmem.eval.metrics import score
from dkmem.eval.ablation import ablation_report
from dkmem.pipeline.cli import DEFAULT_INPUT, load_records
from dkmem.baselines.mem0.source import verify_structure

report = {"checks": {}, "groups": {}}
def check(name, ok, detail=None):
    report["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL "), name, "" if detail is None else str(detail)[:300])

PAIRWISE = json.load(open(os.path.join(SRC, "dkmem/pairwise_eval.schema.json"), encoding="utf-8"))
MANIFEST = json.load(open(os.path.join(SRC, "dkmem/run_manifest.schema.json"), encoding="utf-8"))
RUNS, REP = "/kaggle/working/runs", "/kaggle/working/runs_repeat"
dirs = sorted(d for d in glob.glob(RUNS + "/*") if os.path.isdir(d))
check("groups_present", len(dirs) == 4 + 3 + 1 + 1, [os.path.basename(d) for d in dirs])
expected_prefix = ["A-", "B-", "C-", "D-", "mem0-v1.0.11-", "flat-no-llm-"]
for pre in expected_prefix:
    check("group_" + pre, sum(os.path.basename(d).startswith(pre) for d in dirs) >= 1)
informed = [d for d in dirs if json.load(open(d + "/run_config.json", encoding="utf-8"))["prompts"].get("merge_judge", {}).get("prompt_id") == "merge_judge_informed_v1"]
check("informed_groups", len(informed) == 3, [os.path.basename(d) for d in informed])

# 1. schema validity of every output
bad = []
for d in dirs:
    for mdir in glob.glob(d + "/*/"):
        if os.path.exists(mdir + "run_manifest.json"):
            try:
                jsonschema.validate(json.load(open(mdir + "run_manifest.json", encoding="utf-8")), MANIFEST)
                for line in open(mdir + "pairwise_eval.jsonl", encoding="utf-8"):
                    if line.strip(): jsonschema.validate(json.loads(line), PAIRWISE)
            except Exception as e:
                bad.append((mdir, str(e)[:200]))
check("all_outputs_schema_valid", not bad, bad)

# 2. determinism: same fingerprint, group id, trace and pairwise rows on the repeat
diffs = []
for d in dirs:
    gid = os.path.basename(d)
    r = os.path.join(REP, gid)
    if not os.path.isdir(r):
        if gid.startswith(tuple(["A-", "B-", "C-", "D-", "mem0-", "flat-"])) and gid not in [os.path.basename(i) for i in informed]:
            diffs.append((gid, "no repeat group (fingerprint / group id differs)"))
        continue
    a, b = json.load(open(d + "/run_config.json", encoding="utf-8")), json.load(open(r + "/run_config.json", encoding="utf-8"))
    if a["fingerprint"] != b["fingerprint"]: diffs.append((gid, "fingerprint"))
    for rel in ["trace.jsonl"] + [os.path.relpath(p, d) for p in glob.glob(d + "/*/pairwise_eval.jsonl")]:
        if open(os.path.join(d, rel), "rb").read() != open(os.path.join(r, rel), "rb").read(): diffs.append((gid, rel))
check("repeat_runs_identical", not diffs, diffs)

# 3. provenance of the real model and embedder
cfgs = {os.path.basename(d): json.load(open(d + "/run_config.json", encoding="utf-8")) for d in dirs}
rev_ok, rev_detail = True, {}
for gid, c in cfgs.items():
    mri = c.get("model_run_info")
    if mri:
        rev_detail[gid + ":model"] = (mri["model_config"]["revision"], mri.get("resolved_revision"), mri["model_config"]["dtype"])
        rev_ok &= mri["model_config"]["revision"] == "MODEL_REV" and mri.get("resolved_revision") == "MODEL_REV"
    if c.get("embedder"):
        rev_detail[gid + ":embedder"] = (c["embedder"]["model_config"]["revision"], c["embedder"].get("resolved_revision"))
        rev_ok &= c["embedder"].get("resolved_revision") == "EMB_REV"
    rev_ok &= len(c["code"]["tree_sha256"]) == 64 and c["input"]["n_records"] == LIMIT and c["input"]["sanctioned"] is (not SLICED)
check("model_embedder_revisions_and_code_hash", rev_ok, rev_detail)

# 4. pipeline groups: summaries, failures, gating invariants
for d in dirs:
    gid = os.path.basename(d)
    summ = json.load(open(d + "/summary.json", encoding="utf-8"))
    report["groups"][gid] = summ
    print(gid, json.dumps(summ, ensure_ascii=False)[:700])
for d in dirs:
    gid = os.path.basename(d)
    if not gid[:2] in ("A-", "B-", "C-", "D-"): continue
    c = cfgs[gid]
    trace = [json.loads(l) for l in open(d + "/trace.jsonl", encoding="utf-8")]
    inv = []
    for t in trace:
        if t.get("status") != "ok": continue
        g = t["gate"]
        for side in ("a", "b"):
            if "lexicon_ambiguous" not in t["extraction"][side]: inv.append("lexicon_ambiguous missing")
        if "lexicon+llm" in g and "lexicon" in g and "skipped" not in g["lexicon+llm"]:
            amb = t["extraction"]["a"]["lexicon_ambiguous"] + t["extraction"]["b"]["lexicon_ambiguous"]
            if not amb and (g["lexicon+llm"]["distinction_a"] != g["lexicon"]["distinction_a"] or g["lexicon+llm"]["distinction_b"] != g["lexicon"]["distinction_b"]):
                inv.append(("lexicon+llm differs from lexicon without ambiguity", t["eval_pair_id"]))
        # a gate on can only turn a proposed merge into keep_both / link
        for mode, gm in g.items():
            if "skipped" in gm or not gm.get("applied"): continue
            if gm["vetoed"] and t["host"]["proposed"] not in ("merge", "supersede"): inv.append(("veto without a merge proposal", t["eval_pair_id"]))
        if t["final"]["off"]["decision"] != t["host"]["proposed"]: inv.append(("off != host", t["eval_pair_id"]))
    check("gating_invariants_" + gid, not inv, inv[:5])
    if "lexicon+llm" in c["dkmem_modes"]: check("lexicon_llm_policy_" + gid, c.get("lexicon_llm_policy") == "model_only_for_lexicon_ambiguous_classes_v1")
    else: check("no_lexicon_llm_policy_when_mode_unused_" + gid, "lexicon_llm_policy" not in c)
    n_v4 = sum(1 for t in trace for s in ("a", "b") if t.get("extraction") and t["extraction"][s].get("v4_prompt_id"))
    print(gid, "V4-extracted sides:", n_v4, "| ambiguous sides:", sum(len(t["extraction"][s]["lexicon_ambiguous"]) > 0 for t in trace if t.get("extraction") for s in ("a", "b")))

# 5. Mem0
m = [d for d in dirs if os.path.basename(d).startswith("mem0-")][0]
verify_structure()
mt = [json.loads(l) for l in open(m + "/trace.jsonl", encoding="utf-8")]
check("mem0_pins_verified", True)
check("mem0_raw_io_captured", all(t["add_a"]["extraction"].get("raw_output") is not None and t["add_b"]["extraction"].get("raw_output") is not None for t in mt if "add_a" in t))
print("MEM0 summary", json.dumps(json.load(open(m + "/summary.json", encoding="utf-8")), ensure_ascii=False))
for t in mt[:3]:
    print("MEM0", t["eval_pair_id"], t["status"], t["pair"]["outcome"], t["pair"]["reason"])
    print("   a facts", t["add_a"]["extraction"].get("facts"), "| raw:", (t["add_a"]["extraction"].get("raw_output") or "")[:200].replace("\n", " "))
    print("   b facts", t["add_b"]["extraction"].get("facts"), "| update raw:", ((t["add_b"].get("update") or {}).get("raw_output") or "")[:300].replace("\n", " "))
f = [d for d in dirs if os.path.basename(d).startswith("flat-")][0]
check("flat_never_merges", json.load(open(f + "/summary.json", encoding="utf-8"))["unifying_decisions"] == 0)

# 6. the offline evaluators on the real outputs. PLACEHOLDER gold (alternating same/different): wiring only.
records, _ = load_records(Path("/kaggle/working/slice.jsonl") if SLICED else DEFAULT_INPUT)
records = records[:LIMIT]
gold = gold_from_pairs(GoldPair(r.eval_pair_id, "same" if i % 2 else "different", "kinship", r.language) for i, r in enumerate(records))
groups = [load_group(d) for d in dirs]
dgroup = [g for g in groups if g.config_id == "D"][0]
sw = sweep_config_d(dgroup, gold)
check("sweep_self_check_on_real_D_log", all(v["self_check"]["mismatches"] == [] for v in sw["modes"].values()),
      {m_: (len(v["points"]), v["self_check"]["pairs_checked"]) for m_, v in sw["modes"].items()})
sc_ok = True
for g in groups:
    for mode in g.modes:
        s = score(g.rows(mode), records, gold, no_entry_ids=g.no_entry_ids())
        sc_ok &= s["overall"]["n_different"] + s["overall"]["n_same"] == LIMIT
check("score_runs_on_every_group_and_mode", sc_ok)
ab = ablation_report(dgroup, gold, records)
check("ablation_pairing_violations_empty", ab["pairing_violations"] == [], ab["pairing_violations"][:3])
print("GATE ACTIVITY", {k: v.get("gate") for k, v in ab["modes"].items()})
print("ABLATION extra llm calls:", {k: v["extra_llm_calls"] for k, v in ab["modes"].items()}, "| fallback:", {k: ab["fallback"][k] for k in ("pairs_whose_compatibility_changes", "utterances_whose_distinction_differs")})
fair = check_comparable([g for g in groups if g not in [load_group(i) for i in informed] or True])
report["fairness"] = fair
print("FAIRNESS", json.dumps(fair, ensure_ascii=False, default=str)[:1500])
report["all_pass"] = all(v["ok"] for v in report["checks"].values())
json.dump(report, open("/kaggle/working/smoke_report.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
print("ALL PASS" if report["all_pass"] else "SOME CHECKS FAILED")
'''.replace("MODEL_REV", MODEL_REV).replace("EMB_REV", EMB_REV).replace("LIMIT", str(args.limit)).replace("SLICED_FLAG", str(SLICED))


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md(f"# DK-Mem limit={args.limit} real-model smoke test: A-D, DK-Mem modes, informed judge, Mem0 v1.0.11, flat RAG"),
    code(f"""
import hashlib, json, os, subprocess, sys, time
GPU_LIST = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print(GPU_LIST)
SRC = "/kaggle/working/src"
import base64, io, zipfile
PAYLOAD = "{PAYLOAD}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
    names = z.namelist()
    for rel in names:
        path = os.path.join(SRC, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(z.read(rel))
print(len(names), "files written")
try:
    import jsonschema
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jsonschema"], check=True)
"""),
    code(f"""
SLICE_INCOMPATIBLE = {args.slice_incompatible}
if SLICE_INCOMPATIBLE:
    sys.path.insert(0, SRC)
    from dkmem.pipeline.cli import DEFAULT_INPUT, load_records
    from dkmem.memory.lexicon import load_lexicon
    from dkmem.pipeline.trace import lexicon_distinction
    from dkmem.memory.gate import compatible
    lex = load_lexicon(os.path.join(SRC, "dkmem/memory/distinction_features.json"))
    recs, _ = load_records(DEFAULT_INPUT)
    inc = [r for r in recs if compatible(lexicon_distinction(lex, r.utterance_a, r.language), lexicon_distinction(lex, r.utterance_b, r.language)) == "incompatible"]
    com = [r for r in recs if r not in inc]
    keep = {{r.eval_pair_id for r in inc[:SLICE_INCOMPATIBLE] + com[:{args.limit} - SLICE_INCOMPATIBLE]}}
    with open("/kaggle/working/slice.jsonl", "w", encoding="utf-8", newline="\\n") as f:
        for r in recs:
            if r.eval_pair_id in keep:
                f.write(json.dumps({{k: getattr(r, k) for k in ("eval_pair_id", "language", "utterance_a", "utterance_b", "utterance_a_id", "utterance_b_id", "opaque_entity_id_a", "opaque_entity_id_b")}}, ensure_ascii=False) + "\\n")
    print("slice:", len(keep), "records,", min(SLICE_INCOMPATIBLE, len(inc)), "lexicon-incompatible")
"""),
    code("""
p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=SRC,
                   capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC})
print(p.stderr[-800:])
assert p.returncode == 0, "unit tests failed"
"""),
    code(f"""
COMMANDS = {json.dumps([(n, m, a + ["--out", "/kaggle/working/" + o]) for n, m, a, o in COMMANDS])}
results = {{}}
for name, module, argv in COMMANDS:
    t0 = time.time()
    print("=" * 20, name, "python -m", module, " ".join(argv), flush=True)
    proc = subprocess.Popen([sys.executable, "-m", module, *argv], cwd=SRC, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env={{**os.environ, "PYTHONPATH": SRC}})
    tail = []
    for line in proc.stdout:
        tail.append(line)
        if line.startswith(("config ", "mem0:", "flat:", "{{")) or "Error" in line or "Traceback" in line:
            print(line[:600], end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        print("".join(tail[-40:]))
    results[name] = {{"rc": rc, "seconds": round(time.time() - t0, 1)}}
    print(name, results[name], flush=True)
json.dump({{"nvidia_smi_L": GPU_LIST, "commands": COMMANDS, "results": results}}, open("/kaggle/working/run_environment.json", "w"), indent=2)
print(json.dumps(results))
"""),
    code(CHECKS),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "dkmem_baselines_smoke.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
meta = {"id": f"{args.user}/{args.slug}", "title": args.slug.replace("-", " ").title(), "code_file": "dkmem_baselines_smoke.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
with open(os.path.join(args.out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("wrote", args.out, len(embedded), "files")
