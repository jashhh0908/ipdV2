"""Command line for the offline evaluation (no GPU, no model).

    python -m dkmem.eval.cli score    --group runs/A-... runs/D-... --gold KEY.jsonl [--out score.json]
    python -m dkmem.eval.cli sweep    --group runs/D-...            --gold KEY.jsonl [--out sweep.json]
    python -m dkmem.eval.cli compare  --group runs/A-... runs/store-... runs/mem0-... runs/flat-... \\
                                      --d-group runs/D-... --gold KEY.jsonl [--out compare.json]
    python -m dkmem.eval.cli ablation --group runs/B-...            --gold KEY.jsonl [--out ablation.json]

``--gold`` is Team B's answer key (``dkmem.eval.gold``); Team A does not have it, so these commands are run by whoever
holds it. ``--input`` defaults to the sanctioned Tier 1 file. ``sweep`` is the raise-tau baseline (Config D groups only).
``compare`` puts every system's operating points next to the raise-tau curve at matched MCR and runs the fairness
check (``dkmem.eval.fairness``) over all the groups given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dkmem.eval.ablation import ablation_report
from dkmem.eval.fairness import check_comparable
from dkmem.eval.gold import load_gold
from dkmem.eval.groups import GroupData, load_group
from dkmem.eval.metrics import score
from dkmem.eval.sweep import compare_at_matched_mcr, sweep_config_d
from dkmem.pipeline.cli import DEFAULT_INPUT, load_records

__all__ = ["main"]


def _score_group(group: GroupData, records, gold) -> dict[str, Any]:
    out: dict[str, Any] = {"group_id": group.group_id, "label": group.strategy_label, "tau": group.run_config.get("tau"),
                           "modes": {}}
    for mode in group.modes:
        s = score(group.rows(mode), records, gold, no_entry_ids=group.no_entry_ids())
        out["modes"][mode] = {"strategy": group.manifest(mode)["strategy"], "overall": s["overall"], "breakdown": s["breakdown"]}
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("score", "sweep", "compare", "ablation"))
    ap.add_argument("--group", type=Path, nargs="+", required=True, help="run group directories")
    ap.add_argument("--d-group", type=Path, default=None, help="the Config D group whose gate-off curve is raise-tau")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    records, _ = load_records(args.input)
    gold = load_gold(args.gold)
    groups = [load_group(p) for p in args.group]

    if args.command == "score":
        report: dict[str, Any] = {"groups": [_score_group(g, records, gold) for g in groups]}
    elif args.command == "sweep":
        report = {"sweeps": [sweep_config_d(g, gold) for g in groups]}
    elif args.command == "ablation":
        report = {"ablations": [ablation_report(g, gold, records) for g in groups]}
    else:
        if args.d_group is None:
            ap.error("compare needs --d-group (the Config D group that provides the raise-tau curve)")
        d_group = load_group(args.d_group)
        sweep = sweep_config_d(d_group, gold)
        raise_tau = sweep["modes"]["off"]["points"] if "off" in sweep["modes"] else ap.error("the D group has no 'off' mode")
        everyone = [d_group, *[g for g in groups if g.path.resolve() != d_group.path.resolve()]]
        scored = [_score_group(g, records, gold) for g in everyone]
        for entry in scored:
            for mode, m in entry["modes"].items():
                point = {"tau": entry["tau"], "fcr": m["overall"]["fcr"], "mcr": m["overall"]["mcr"]}
                m["vs_raise_tau"] = compare_at_matched_mcr([point], raise_tau)[0] if point["fcr"] is not None and point["mcr"] is not None else None
        report = {
            "fairness": check_comparable(everyone),
            "raise_tau": {"group_id": d_group.group_id, "n_scores": sweep["n_scores"], "points": raise_tau,
                          "frontier": sweep["modes"]["off"]["frontier"]},
            "dkmem_on_d": {m: v["frontier"] for m, v in sweep["modes"].items() if m != "off"},
            "groups": scored,
        }
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
