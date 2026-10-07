"""Compare the candidate Config B prompt with the current one from native_b_comparison.json.

Usage: python analyze.py <kaggle_output_dir> [metrics.json] [side_by_side.md]
"""

import difflib
import glob
import json
import os
import re
import sys
from collections import Counter

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def words(t):
    return [w for w in re.split(r"[^\w]+", t.casefold()) if w]


def metrics(name, o, smoke_dir=None):
    sides = o["sides"]
    n = len(sides)
    with_fact = [x for x in sides if x["facts"]]
    facts = [(x, f, r) for x in sides for f, r in zip(x["facts"], x["language_reports"])]
    out = {"variant": name, "sides": n,
           "parsed_ok": sum(1 for x in sides if x["error"] is None),
           "parse_failed": sum(1 for x in sides if x["error"] is not None),
           "sides_with_a_fact": len(with_fact), "empty_fact_list": sum(1 for x in sides if x["error"] is None and not x["facts"]),
           "sides_with_exactly_one_fact": sum(1 for x in sides if len(x["facts"]) == 1),
           "facts_total": len(facts)}
    out["parsed_rate"] = round(out["parsed_ok"] / n, 3)
    out["one_fact_rate"] = round(out["sides_with_exactly_one_fact"] / n, 3)
    out["language_label"] = dict(Counter(r["label"] for _, _, r in facts))
    ret = [r["source_token_retention"] for _, _, r in facts if r["source_token_retention"] is not None]
    out["mean_source_token_retention"] = round(sum(ret) / len(ret), 3) if ret else None
    out["verbatim_copy"] = sum(1 for x, f, _ in facts if f.strip() == x["utterance"].strip())
    out["near_copy_ge_0.9"] = sum(1 for x, f, _ in facts
                                  if difflib.SequenceMatcher(None, f.casefold(), x["utterance"].casefold()).ratio() >= 0.9)
    dev = [(x, f) for x, f, _ in facts if DEVANAGARI.search(x["utterance"])]
    out["devanagari_utterances_with_fact"] = len(dev)
    out["devanagari_kept_in_fact"] = sum(1 for _, f in dev if DEVANAGARI.search(f))
    pairs = o["pairs"]
    out["pairs_judged"] = len(pairs)
    out["judge_parse_ok"] = sum(1 for p in pairs if p["judge_error"] is None)
    out["judge_decisions"] = dict(Counter(p["judge_decision"] for p in pairs))
    out["gate_decisions_lexicon"] = dict(Counter(p.get("gate_decision") for p in pairs))
    out["first_loss_point"] = dict(Counter(p.get("first_loss_point") for p in pairs))
    out["seconds"] = o["seconds"]
    return out


def main(d, metrics_path=None, md_path=None):
    data = json.load(open(os.path.join(d, "native_b_comparison.json"), encoding="utf-8"))
    outs = data["outputs"]
    res = {"variants": [metrics(k, v) for k, v in outs.items()], "prompts": data["prompts"],
           "peak_gpu_memory_gb": data["peak_gpu_memory_gb"], "nvidia_smi_L": data["nvidia_smi_L"],
           "revision": data["run_info"]["resolved_revision"]}

    # does the current-B rerun reproduce the smoke-test outputs?
    smoke = glob.glob(os.path.join(os.path.dirname(d), "..", "smoke_abc", "kaggle_output", "runs", "B-*", "trace.jsonl"))
    if smoke:
        sm = {}
        for l in open(smoke[0], encoding="utf-8"):
            t = json.loads(l)
            for s in ("a", "b"):
                sm[(t["eval_pair_id"], s)] = t["extraction"][s]["raw_output"]
        cur = {(x["eval_pair_id"], x["side"]): x["raw_output"] for x in outs["current_B_mem0_legacy"]["sides"]}
        res["current_B_matches_smoke_raw_outputs"] = {"matching": sum(1 for k in cur if sm.get(k) == cur[k]), "of": len(cur)}

    cand, cur = outs["candidate_B_native"], outs["current_B_mem0_legacy"]
    lines = ["| pair | side | utterance | current B fact | candidate B fact | candidate label | retention |", "|---|---|---|---|---|---|---|"]
    cmap = {(x["eval_pair_id"], x["side"]): x for x in cur["sides"]}
    for x in cand["sides"]:
        c = cmap[(x["eval_pair_id"], x["side"])]
        rep = x["language_reports"][0] if x["language_reports"] else {}
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            x["eval_pair_id"][-4:], x["side"], x["utterance"],
            " / ".join(c["facts"]) or "(none)", " / ".join(x["facts"]) or f"ERR {x['error']}",
            rep.get("label", "-"), rep.get("source_token_retention", "-")))
    lines += ["", "| pair | candidate entry A | candidate entry B | judge | gate (lexicon) |", "|---|---|---|---|---|"]
    for p in cand["pairs"]:
        lines.append(f"| {p['eval_pair_id'][-4:]} | {p['entry_a']} | {p['entry_b']} | {p['judge_decision']} | {p.get('gate_decision')} |")
    if metrics_path:
        json.dump(res, open(metrics_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if md_path:
        open(md_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:4]))
