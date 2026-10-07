"""Parse, classify and tabulate the candidate Config B outputs (3 models x 346 sides).

Two stages:
  python analysis.py prepare   -> parsed.json, to_label.jsonl  (the unique non-copy outputs to read by hand)
  python analysis.py report    -> metrics.json                 (needs hand_labels.json)
"""

import difflib
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from dkmem.memory.native_extract import output_language_report  # noqa: E402  (v2 classifier)
from dkmem.memory.lexicon import load_lexicon  # noqa: E402
from dkmem.pipeline.trace import lexicon_distinction  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "kaggle_output")
MODELS = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]
LEX = load_lexicon(os.path.join(REPO, "dkmem", "memory", "distinction_features.json"))

# Relationship terms in the utterances (Latin script and Devanagari, plus the languages of the data).
KIN_WORDS = set("""
chacha chachi chachaji chacha-ji mama mami mamaji mausi mausa mausaji bua fufa fufaji nana nani nanaji dada dadi dadaji
bhaiya bhai bhabhi didi behen behan beta beti papa mummy mummyji pitaji mataji uncle aunty auntie cousin jija jijaji sala
saali devar jeth nanad sasur saas dost
bruder schwester mutter vater onkel tante oma opa
abla abi amca teyze hala dayi dayı anne baba kardeş
""".split()) - {"dost"}
KIN_DEVANAGARI = ["चाचा", "चाची", "मामा", "मामी", "मौसी", "मौसा", "बुआ", "फूफा", "नाना", "नानी", "दादा", "दादी", "भैया", "भाभी", "दीदी", "बेटा", "बेटी"]
TRAILING = re.compile(r"[\s.?!।]+$")


def v1_report(source, stored, lang):
    """The previous classifier (v1), kept here only to measure the loophole fix."""
    from dkmem.memory.extract import _ENGLISH_WORDS
    import dkmem.memory.native_extract as ne
    ws = lambda t: [w for w in re.split(r"[^\w]+", t.casefold()) if w and not w.isdigit()]
    out_scripts, src_scripts = ne.script_shares(stored), ne.script_shares(source)
    if lang.lower() == "en" or not out_scripts or not src_scripts:
        return "not_applicable"
    dom = max(src_scripts, key=src_scripts.get)
    if dom != "LATIN":
        share = out_scripts.get(dom, 0.0)
        return "source_language" if share >= 0.5 else "translated_to_english" if share <= 0.05 else "mixed"
    stored_words = set(ws(stored))
    foreign = [w for w in ws(source) if w not in _ENGLISH_WORDS]
    if not foreign:
        return "not_applicable"
    r = sum(1 for w in foreign if w in stored_words) / len(foreign)
    return "source_language" if r >= 0.5 else "translated_to_english" if r <= 0.2 else "mixed"


def norm(t):
    return re.sub(r"\s+", " ", TRAILING.sub("", t.strip())).casefold()


def parse(raw):
    try:
        d = json.loads(raw)
    except Exception as e:
        return None, f"not a single JSON object ({e})"
    if not isinstance(d, dict) or set(d) != {"fact"}:
        return None, f"keys must be exactly ['fact'], got {d!r}"
    if not isinstance(d["fact"], str) or not d["fact"].strip():
        return None, "fact must be a non-empty string"
    return d["fact"], None


def kin_terms(utt, lang):
    toks = [t.casefold() for t in re.findall(r"[^\W\d_]+", utt)]
    found = {t for t in toks if t in KIN_WORDS}
    found |= {k for k in KIN_DEVANAGARI if k in utt}
    return sorted(found)


def multi_sentence(fact):
    return len(re.findall(r"[.?!।]\s+\S", fact.strip())) > 0


def prepare():
    rows = []
    for m in MODELS:
        d = json.load(open(os.path.join(OUT, f"native_b_{m}.json"), encoding="utf-8"))
        for s in d["sides"]:
            fact, err = parse(s["raw_output"])
            row = {"model": m, **{k: s[k] for k in ("eval_pair_id", "side", "language", "utterance")},
                   "raw_output": s["raw_output"], "fact": fact, "error": err}
            if fact is not None:
                row["exact_copy"] = fact == s["utterance"]
                row["norm_copy"] = norm(fact) == norm(s["utterance"])
                row["multi_sentence"] = multi_sentence(fact)
                rep = output_language_report(s["utterance"], fact, s["language"])
                row["v2"] = rep["label"]
                row["v2_report"] = {k: rep[k] for k in ("source_token_retention", "content_words", "english_function_share", "source_script_share")}
                row["v1"] = v1_report(s["utterance"], fact, s["language"])
                kt = kin_terms(s["utterance"], s["language"])
                row["kin_terms"] = kt
                lex = lexicon_distinction(LEX, s["utterance"], s["language"])
                row["lexicon_kinship"] = lex.get("kinship")
            rows.append(row)
    json.dump(rows, open(os.path.join(HERE, "parsed.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    uniq = {}
    for r in rows:
        if r["fact"] is not None and not r["norm_copy"]:
            uniq.setdefault((r["utterance"], r["fact"]), r["language"])
    items = [{"id": i, "utterance": u, "fact": f, "language": l} for i, ((u, f), l) in enumerate(sorted(uniq.items(), key=lambda kv: (kv[0][0], kv[0][1])))]
    with open(os.path.join(HERE, "to_label.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = Counter(r["model"] for r in rows if r["fact"] is not None and not r["norm_copy"])
    print("sides", len(rows), "| parse failures", sum(1 for r in rows if r["fact"] is None),
          "| non-copy sides per model", dict(c), "| unique non-copy (utterance, fact) to label:", len(items))


if __name__ == "__main__":
    {"prepare": prepare}[sys.argv[1]]()
