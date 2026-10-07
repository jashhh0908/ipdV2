"""Score extraction_validation_<prompt_id>.json files from the Kaggle run.

Criteria (identical for every prompt; the first two are those of the earlier
v1/v2/v3 comparison):
- target-feature correct: the expected {feature: value} is present in the output
  (extra keys ignored); for an unmarked side, the probe's feature is absent.
- exact match: the output distinction equals the expected distinction.
- in-scope: probes of class kinship, register, name_variant or control_kinship
  (12 of the 20 sides); the rest target the dropped classes.
- outcome per side: correct / wrong_value / missing / over_marked / invalid.
- spurious keys: keys in a valid output that are not the expected key; of these,
  "out-of-scope" ones are politeness / evidentiality / classifier / temporal_deixis.
  Rejected outputs are counted separately from their diagnostic parse.
- gloss violations: the gloss contains I / me / my / you (any case, whole word).

Usage: python score.py <dir> [out.json]
"""

import json
import os
import re
import sys

IN_SCOPE_PROBE_CLASSES = {"kinship", "register", "name_variant", "control_kinship"}
OUT_OF_SCOPE_KEYS = {"politeness", "evidentiality", "classifier", "temporal_deixis"}
YOU_RE = re.compile(r"\b(?:i|me|my|you)\b", re.IGNORECASE)
PROMPTS = ["mem0_extraction_v2", "mem0_extraction_v3", "mem0_extraction_v4"]


def feature_of(row):
    return row["distinction_class"].removeprefix("control_")


def outcome(row):
    if not row["success"]:
        return "invalid"
    exp, got = row["expected_distinction"], row["distinction"]
    if exp:
        ((k, v),) = exp.items()
        if got.get(k) == v:
            return "correct"
        return "wrong_value" if k in got else "missing"
    return "correct" if feature_of(row) not in got else "over_marked"


def score(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = d["items"]
    out = {"prompt_id": d["prompt_id"], "prompt_sha256": d["prompt_sha256"], "n": len(rows)}
    oc = [outcome(r) for r in rows]
    out["target_correct"] = oc.count("correct")
    out["exact"] = sum(1 for r in rows if r["success"] and r["distinction"] == r["expected_distinction"])
    ins = [i for i, r in enumerate(rows) if r["distinction_class"] in IN_SCOPE_PROBE_CLASSES]
    out["in_scope_n"] = len(ins)
    out["in_scope_correct"] = sum(1 for i in ins if oc[i] == "correct")
    out["in_scope_exact"] = sum(
        1 for i in ins if rows[i]["success"] and rows[i]["distinction"] == rows[i]["expected_distinction"]
    )
    out["out_of_scope_target_correct"] = sum(1 for i, r in enumerate(rows) if i not in ins and oc[i] == "correct")
    out["outcomes"] = {k: oc.count(k) for k in ("correct", "wrong_value", "missing", "over_marked", "invalid")}
    out["in_scope_outcomes"] = {k: sum(1 for i in ins if oc[i] == k)
                                for k in ("correct", "wrong_value", "missing", "over_marked", "invalid")}

    spurious, oos_valid = [], []
    for r in rows:
        if not r["success"]:
            continue
        for k in r["distinction"]:
            if k not in r["expected_distinction"]:
                spurious.append((r["pair_id"], r["side"], k, r["distinction"][k]))
            if k in OUT_OF_SCOPE_KEYS:
                oos_valid.append((r["pair_id"], r["side"], k))
    out["spurious_keys_valid_outputs"] = len(spurious)
    out["spurious_in_scope_keys"] = sum(1 for s in spurious if s[2] not in OUT_OF_SCOPE_KEYS)
    out["spurious_out_of_scope_keys"] = sum(1 for s in spurious if s[2] in OUT_OF_SCOPE_KEYS)
    out["sides_with_spurious_key"] = len({(s[0], s[1]) for s in spurious})
    out["out_of_scope_keys_in_valid_outputs"] = len(oos_valid)
    out["spurious_detail"] = [list(s) for s in spurious]

    rejected_oos = []
    for r in rows:
        if r["success"] or not isinstance(r["distinction"], dict):
            continue
        rejected_oos += [(r["pair_id"], r["side"], k) for k in r["distinction"] if k in OUT_OF_SCOPE_KEYS]
    out["out_of_scope_keys_in_rejected_outputs"] = len(rejected_oos)

    viol = [(r["pair_id"], r["side"], r["gloss"]) for r in rows
            if isinstance(r["gloss"], str) and YOU_RE.search(r["gloss"])]
    out["gloss_violations"] = len(viol)
    out["gloss_violation_detail"] = [list(v) for v in viol]

    bad = [r for r in rows if not r["success"]]
    out["invalid_or_failed"] = len(bad)
    out["failure_categories"] = d["summary"]["failure_categories"]
    out["failure_detail"] = [[r["pair_id"], r["side"], r["failure_category"], r["error"]] for r in bad]
    out["surface_mismatches"] = sum(1 for r in rows if r["surface_exact_match"] is False)
    return out


def markdown(results):
    cols = [r["prompt_id"].replace("mem0_extraction_", "").upper() for r in results]
    lines = ["| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(str(fn(r)) for r in results) + " |")

    row("Correct distinctions (target-feature), of 20", lambda r: f"{r['target_correct']}/20")
    row("Exact-match distinctions, of 20", lambda r: f"{r['exact']}/20")
    row("Correct in-scope distinctions (kinship / register / name_variant), of 12",
        lambda r: f"{r['in_scope_correct']}/{r['in_scope_n']}")
    row("In-scope exact match, of 12", lambda r: f"{r['in_scope_exact']}/{r['in_scope_n']}")
    row("Correct on out-of-scope probe sides, of 8", lambda r: f"{r['out_of_scope_target_correct']}/8")
    row("Wrong value (expected key present, wrong value)", lambda r: r["outcomes"]["wrong_value"])
    row("Missing (expected key absent)", lambda r: r["outcomes"]["missing"])
    row("Over-marked (unmarked side given a key)", lambda r: r["outcomes"]["over_marked"])
    row("Invalid / extraction failure", lambda r: r["outcomes"]["invalid"])
    row("Spurious keys in valid outputs (any key not expected)", lambda r: r["spurious_keys_valid_outputs"])
    row("  of which out-of-scope class keys", lambda r: r["spurious_out_of_scope_keys"])
    row("  of which in-scope keys (wrong place / unmarked side)", lambda r: r["spurious_in_scope_keys"])
    row("Out-of-scope keys in rejected outputs", lambda r: r["out_of_scope_keys_in_rejected_outputs"])
    row("`you` / `My...` gloss violations", lambda r: r["gloss_violations"])
    row("Invalid JSON / extraction failures", lambda r: r["invalid_or_failed"])
    row("Surface mismatches", lambda r: r["surface_mismatches"])
    return "\n".join(lines)


def main(directory, out_path=None):
    results = [score(os.path.join(directory, f"extraction_validation_{p}.json")) for p in PROMPTS]
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(markdown(results))
    return results


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
