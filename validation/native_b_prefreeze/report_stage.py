"""Tabulate the hand-labelled candidate Config B outputs and validate the language classifier.

Run after `python analysis.py prepare` and the hand labelling (hand_labels.txt):
    python report_stage.py        -> metrics.json, labeled_sides.json
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A  # noqa: E402
from dkmem.memory.native_extract import output_language_report, script_shares  # noqa: E402

HERE = A.HERE
MODELS = A.MODELS
KIN_KOREAN = ["동생", "따님", "딸", "부인", "아내", "아드님", "아들", "아빠", "엄마", "형", "할머니"]
COARSE = ["source_language", "translated_to_english", "mixed", "script_changed", "undetermined"]


def has_kin(utt):
    toks = [t.casefold() for t in re.findall(r"[^\W\d_]+", utt)]
    return any(t in A.KIN_WORDS for t in toks) or any(k in utt for k in A.KIN_DEVANAGARI + KIN_KOREAN)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 3), round(c + h, 3)]


def rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def load_hand():
    items = {i["id"]: (i["utterance"], i["fact"]) for i in map(json.loads, open(os.path.join(HERE, "to_label.jsonl"), encoding="utf-8"))}
    labels = {}
    for line in open(os.path.join(HERE, "hand_labels.txt"), encoding="utf-8"):
        i, lang, corr, kin = line.split()
        labels[items[int(i)]] = {"lang": lang, "corr": "" if corr == "-" else corr, "kin": "" if kin == "-" else kin}
    assert len(labels) == len(items) == 552
    return labels


def coarse_hand(hl, utterance):
    shares = script_shares(utterance)
    src_latin = max(shares, key=shares.get) == "LATIN"
    return {"S": "source_language", "E": "translated_to_english", "O": "translated_to_english", "M": "mixed",
            "T": "script_changed", "N": "script_changed" if src_latin else "translated_to_english"}[hl["lang"]]


def confusion(rows, key):
    mat = defaultdict(Counter)
    for r in rows:
        mat[r["hand_coarse"]][r[key]] += 1
    n = sum(sum(c.values()) for c in mat.values())
    acc = sum(mat[h][h] for h in mat) / max(1, n)
    return {"accuracy": round(acc, 3), "n": n, "hand_rows_classifier_cols": {h: dict(c) for h, c in mat.items()}}


def main():
    rows = json.load(open(os.path.join(HERE, "parsed.json"), encoding="utf-8"))
    hand = load_hand()
    inconsistent = []
    for r in rows:
        if r["fact"] is None:
            continue
        r["kin_in_utt"] = has_kin(r["utterance"])
        if r["norm_copy"]:
            h = {"lang": "S", "corr": "", "kin": "k" if r["kin_in_utt"] else ""}
        else:
            h = dict(hand[(r["utterance"], r["fact"])])
            if bool(h["kin"]) != r["kin_in_utt"]:
                inconsistent.append([r["utterance"], r["fact"], h["kin"], r["kin_in_utt"]])
        r["hand"] = h
        r["hand_coarse"] = coarse_hand(h, r["utterance"])
        r["v2"] = output_language_report(r["utterance"], r["fact"], r["language"])["label"]
    res = {"models": {}, "kin_flag_inconsistencies_hand_vs_auto": inconsistent}
    for m in MODELS:
        rr = [r for r in rows if r["model"] == m]
        ok = [r for r in rr if r["fact"] is not None]
        n = len(rr)
        md = {"sides": n, "parsed": rate(len(ok), n),
              "exactly_one_fact": rate(sum(1 for r in ok if not r["multi_sentence"]), n),
              "exact_copy": rate(sum(r["exact_copy"] for r in ok), n),
              "copy_ignoring_case_punct_space": rate(sum(r["norm_copy"] for r in ok), n)}
        hl = Counter(r["hand"]["lang"] for r in ok)
        md["hand_language_counts"] = dict(hl)
        md["language_drift_E_O_N"] = rate(hl["E"] + hl["O"] + hl["N"], n)
        md["language_drift_to_English_E"] = rate(hl["E"], n)
        md["script_drift_same_language_T"] = rate(hl["T"], n)
        md["mixed_M"] = rate(hl["M"], n)
        md["source_language_and_script_kept_S"] = rate(hl["S"], n)
        md["any_departure_from_instruction"] = rate(n - hl["S"], n)
        cc = Counter()
        for r in ok:
            for ch in set(r["hand"]["corr"]):
                cc[ch] += 1
        md["corruption_any"] = rate(sum(1 for r in ok if r["hand"]["corr"]), n)
        md["corruption_types_count"] = {k: cc.get(k, 0) for k in "gimo"}
        md["corruption_excluding_omission_only"] = rate(sum(1 for r in ok if set(r["hand"]["corr"]) & set("gim")), n)
        cross = Counter()
        for r in ok:
            grp = {"S": "source", "T": "script_changed", "M": "mixed"}.get(r["hand"]["lang"], "language_drift")
            cross[f"{grp}|{'corrupt' if r['hand']['corr'] else 'clean'}"] += 1
        md["drift_x_corruption"] = dict(sorted(cross.items()))
        md["classifier_v2_label_rates"] = {k: rate(sum(1 for r in ok if r["v2"] == k), n) for k in COARSE}
        md["classifier_v1_label_rates"] = {k: rate(sum(1 for r in ok if r["v1"] == k), n) for k in COARSE}
        kin = [r for r in ok if r["kin_in_utt"]]
        md["kinship_sides"] = len(kin)
        md["kinship_preserved_as_written"] = rate(sum(1 for r in kin if r["hand"]["kin"] == "k"), len(kin))
        lexk = [r for r in kin if r.get("lexicon_kinship")]
        md["lexicon_kinship_sides"] = len(lexk)
        md["lexicon_kinship_preserved"] = rate(sum(1 for r in lexk if r["hand"]["kin"] == "k"), len(lexk))
        md["kinship_lost_by_cause"] = dict(Counter(
            "translated" if r["hand"]["lang"] in "EON" else "garbled" if "g" in r["hand"]["corr"] else "other"
            for r in kin if r["hand"]["kin"] == "l"))
        bl = {}
        for lg in sorted({r["language"] for r in rr}):
            sub = [r for r in ok if r["language"] == lg]
            h = Counter(r["hand"]["lang"] for r in sub)
            bl[lg] = {"sides": len(sub), "source_kept_S": h["S"], "language_drift_E_O_N": h["E"] + h["O"] + h["N"],
                      "script_changed_T": h["T"], "mixed_M": h["M"],
                      "corrupt": sum(1 for r in sub if r["hand"]["corr"]), "exact_copy": sum(r["exact_copy"] for r in sub)}
        md["by_language"] = bl
        md["format_leaks_Utterance_prefix"] = sum(1 for r in ok if r["fact"].startswith("Utterance:"))
        res["models"][m] = md

    allok = [r for r in rows if r["fact"] is not None]
    noncopy = [r for r in allok if not r["norm_copy"]]
    res["classifier_validation"] = {
        "v2_all_sides": confusion(allok, "v2"), "v1_all_sides": confusion(allok, "v1"),
        "v2_non_copy_sides": confusion(noncopy, "v2"), "v1_non_copy_sides": confusion(noncopy, "v1"),
        "v2_non_copy_per_model_accuracy": {m: confusion([r for r in noncopy if r["model"] == m], "v2")["accuracy"] for m in MODELS},
        "v1_non_copy_per_model_accuracy": {m: confusion([r for r in noncopy if r["model"] == m], "v1")["accuracy"] for m in MODELS},
    }
    res["classifier_rate_error"] = {}
    for m in MODELS:
        ok = [r for r in allok if r["model"] == m]
        res["classifier_rate_error"][m] = {
            "n": len(ok),
            "translated": {"hand": sum(1 for r in ok if r["hand_coarse"] == "translated_to_english"),
                           "v2": sum(1 for r in ok if r["v2"] == "translated_to_english"),
                           "v1": sum(1 for r in ok if r["v1"] == "translated_to_english")},
            "not_source": {"hand": sum(1 for r in ok if r["hand_coarse"] != "source_language"),
                           "v2": sum(1 for r in ok if r["v2"] != "source_language"),
                           "v1": sum(1 for r in ok if r["v1"] != "source_language")}}
    res["v2_disagreements"] = [
        {"model": r["model"], "utterance": r["utterance"], "fact": r["fact"], "hand": r["hand"]["lang"],
         "hand_coarse": r["hand_coarse"], "v2": r["v2"], "v1": r["v1"]}
        for r in noncopy if r["v2"] != r["hand_coarse"]]
    json.dump(res, open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    slim = [{k: v for k, v in r.items() if k != "v2_report"} for r in rows]
    json.dump(slim, open(os.path.join(HERE, "labeled_sides.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    out = main()
    print(json.dumps({m: {k: v for k, v in d.items() if k not in ("by_language", "classifier_v1_label_rates")}
                      for m, d in out["models"].items()}, ensure_ascii=False, indent=1))
