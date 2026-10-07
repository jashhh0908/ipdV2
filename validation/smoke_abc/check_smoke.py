"""Check the Config A-C smoke-test output: parse/judge rates, B language, trace completeness.

Usage: python check_smoke.py <kaggle_output_dir> [out.json]
Read-only on the outputs; prints a summary and optionally writes it as JSON.
"""

import glob
import json
import os
import sys
from collections import Counter

import jsonschema

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAIRWISE_SCHEMA = json.load(open(os.path.join(REPO, "dkmem", "pairwise_eval.schema.json"), encoding="utf-8"))
MANIFEST_SCHEMA = json.load(open(os.path.join(REPO, "dkmem", "run_manifest.schema.json"), encoding="utf-8"))
MODES = {"off": "off", "lexicon": "lexicon", "lexicon+llm": "lexicon-llm"}
TOP_KEYS = {"schema", "group_id", "config", "status", "eval_pair_id", "language", "input"}
OK_KEYS = TOP_KEYS | {"record_id", "extraction", "stored", "similarity", "host", "gate", "final", "diagnosis"}


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def check_config(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "run_config.json"), encoding="utf-8"))
    cid = cfg["pipeline_config"]["config_id"]
    trace = jl(os.path.join(run_dir, "trace.jsonl"))
    out = {"config": cid, "group_id": cfg["group_id"], "trace_rows": len(trace)}
    out["status"] = dict(Counter(t["status"] for t in trace))
    ok = [t for t in trace if t["status"] == "ok"]

    # --- extraction / parse success -------------------------------------------------
    sides = [(t, s) for t in trace for s in ("a", "b") if "extraction" in t]
    host_ext = Counter()
    v4_ext = Counter()
    seen = set()
    for t, s in sides:
        key = (t["eval_pair_id"], s)
        if key in seen:
            continue
        seen.add(key)
        e = t["extraction"][s]
        host_ext[e["status"]] += 1
        if e.get("v4_prompt_id") or e.get("v4_error"):
            v4_ext["failed" if e.get("v4_error") else "ok"] += 1
    out["host_extraction_sides"] = dict(host_ext)
    out["v4_extraction_sides"] = dict(v4_ext)
    out["unique_sides_in_trace"] = len(seen)

    # --- judge ----------------------------------------------------------------------
    judged = [t for t in trace if (t.get("host") or {}).get("judge")]
    jerr = [t for t in judged if t["host"]["judge"]["error"]]
    out["judge_calls"] = len(judged)
    out["judge_parse_ok"] = len(judged) - len(jerr)
    out["judge_decisions"] = dict(Counter(t["host"]["judge"]["decision"] for t in judged if not t["host"]["judge"]["error"]))
    out["judge_errors"] = [{"pair": t["eval_pair_id"], "raw": t["host"]["judge"]["raw_output"],
                            "error": t["host"]["judge"]["error"]} for t in jerr]

    # --- Config B native extraction ---------------------------------------------------
    if cid == "B":
        nat = [(t["extraction"][s], t["stored"][s]) for t in ok for s in ("a", "b")]
        out["B_labels"] = dict(Counter(st["output_language"]["label"] for _, st in nat))
        out["B_fence_stripped"] = sum(1 for e, _ in nat if e.get("fence_stripped"))
        facts_per_side = Counter()
        for t in trace:
            if "extraction" in t:
                for s in ("a", "b"):
                    facts_per_side[(t["eval_pair_id"], s)] = t["extraction"][s]["n_entries"]
        out["B_facts_per_side"] = dict(Counter(facts_per_side.values()))
        out["B_native_failures"] = [{"pair": t["eval_pair_id"], "side": s, "raw": t["extraction"][s]["raw_output"],
                                     "error": t["extraction"][s]["error"]}
                                    for t in trace if t["status"] == "extraction_failed" for s in ("a", "b")
                                    if t["extraction"][s]["status"] == "extraction_failed"]
        out["B_examples"] = [{"utt": t["input"]["utterance_a"], "stored": t["stored"]["a"]["text"],
                              "label": t["stored"]["a"]["output_language"]["label"],
                              "retention": t["stored"]["a"]["output_language"]["source_token_retention"]}
                             for t in ok[:8]]

    # --- completeness -----------------------------------------------------------------
    problems = []
    for t in trace:
        need = OK_KEYS if t["status"] == "ok" else TOP_KEYS
        missing = need - set(t)
        if missing:
            problems.append((t["eval_pair_id"], "missing keys", sorted(missing)))
        if t["status"] == "ok":
            if set(t["gate"]) != set(MODES) or set(t["final"]) != set(MODES):
                problems.append((t["eval_pair_id"], "modes missing in gate/final"))
            if not t["input"].get("input_sha256") or not t["extraction"]["a"]["mode"]:
                problems.append((t["eval_pair_id"], "input hash/extraction mode missing"))
            ex = t["extraction"]
            needs_raw = cid in ("A", "B")
            for s in ("a", "b"):
                if needs_raw and not ex[s]["raw_output"]:
                    problems.append((t["eval_pair_id"], f"raw_output missing side {s}"))
            if t["host"]["judge"] is None or not t["host"]["judge"]["raw_output"]:
                problems.append((t["eval_pair_id"], "judge raw missing"))
            for k in ("score", "metric", "text_a", "text_b"):
                if t["similarity"].get(k) in (None, ""):
                    problems.append((t["eval_pair_id"], f"similarity.{k} missing"))
            for k in ("first_loss_point", "lexicon_compatibility", "host_proposed"):
                if t["diagnosis"].get(k) is None:
                    problems.append((t["eval_pair_id"], f"diagnosis.{k} missing"))
    out["trace_problems"] = problems

    # per-mode files vs trace
    modes = {}
    for mode, slug in MODES.items():
        d = os.path.join(run_dir, slug)
        rows = jl(os.path.join(d, "pairwise_eval.jsonl"))
        man = json.load(open(os.path.join(d, "run_manifest.json"), encoding="utf-8"))
        jsonschema.validate(man, MANIFEST_SCHEMA)
        for r in rows:
            jsonschema.validate(r, PAIRWISE_SCHEMA)
        traced = {t["final"][mode]["record_id"] for t in ok if t["final"].get(mode)}
        modes[mode] = {
            "rows": len(rows), "schema_valid": True,
            "rows_match_trace": traced == {r["record_id"] for r in rows},
            "decisions": dict(Counter(r["decision"] for r in rows)),
            "compat": dict(Counter(str(r["compatibility"]) for r in rows)),
            "strategy": man["strategy"], "manifest_run_id_matches_rows": all(r["run_id"] == man["run_id"] for r in rows),
        }
    out["modes"] = modes
    out["gate_vetoes"] = {m: sum(1 for t in ok if (t["gate"].get(m) or {}).get("vetoed")) for m in ("lexicon", "lexicon+llm")}
    out["host_proposed"] = dict(Counter(t["host"]["proposed"] for t in ok))
    out["first_loss_point"] = dict(Counter(t["diagnosis"]["first_loss_point"] for t in ok))
    out["lexicon_compatibility"] = dict(Counter(t["diagnosis"]["lexicon_compatibility"] for t in ok))
    out["similarity_range"] = [min(t["similarity"]["score"] for t in ok), max(t["similarity"]["score"] for t in ok)] if ok else None
    out["run_config_has"] = {k: (k in cfg and cfg[k] is not None) for k in
                             ("seed", "backbone", "model_run_info", "prompts", "lexicon", "input", "fingerprint", "git")}
    out["input_hash"] = cfg["input"]["inputs_sha256"]
    return out


def main(d, out_path=None):
    results = [check_config(p) for p in sorted(glob.glob(os.path.join(d, "runs", "*")))]
    det = json.load(open(os.path.join(d, "determinism.json"), encoding="utf-8"))
    res = {"configs": results, "determinism": det["config_A_rerun"], "stats": {k: {kk: vv for kk, vv in v.items() if kk != "paths"} for k, v in det["stats"].items()},
           "peak_gpu_memory_gb": det["peak_gpu_memory_gb"]}
    if out_path:
        json.dump(res, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
