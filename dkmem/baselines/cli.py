"""Command line for the non-A-D baselines (needs a GPU for real runs).

Mem0 OSS v1.0.11 with the run's Qwen model and bge-m3 (the same model, revision, decoding and embedder as the
Config A-D runs it is compared with)::

    python -m dkmem.baselines.cli mem0 --model-id Qwen/Qwen2.5-3B-Instruct \\
        --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \\
        --embedding-revision 5617a9f61b028005a4858fdac845db406aefb181 --seed 0 --out runs/

Never-merge / flat dense RAG (no LLM)::

    python -m dkmem.baselines.cli flat --embedding-revision 5617a9f61b028005a4858fdac845db406aefb181 --out runs/

Both write one group directory under ``--out`` (``mem0-v1.0.11-...`` / ``flat-no-llm-...``) in the layout of an A-D
group (``off/`` only). The prompt-informed judge is the A-D CLI with ``--judge-prompt merge_judge_informed_v1
--modes off``; raise-tau is ``python -m dkmem.eval.cli sweep`` on a Config D group.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from dkmem.backends.llm import GenerationParams
from dkmem.baselines.flat import build_flat_run_config, run_flat_baseline, write_flat_outputs
from dkmem.baselines.mem0.runner import build_mem0_run_config, run_mem0_baseline, write_mem0_outputs
from dkmem.baselines.mem0.source import verify_structure
from dkmem.pipeline.cli import DEFAULT_INPUT, default_load_embedder, default_load_generator, load_records

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", choices=("mem0", "flat"))
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", action="store_true", help="split the LLM over the visible GPUs (Qwen2.5-7B)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8, help="recorded; Mem0 itself calls the LLM one prompt at a time")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N records (smoke test)")
    ap.add_argument("--embedding-model-id", default="BAAI/bge-m3")
    ap.add_argument("--embedding-revision", default=None)
    ap.add_argument("--k", type=int, default=5, help="flat: write-time retrieval depth (irrelevant to the result)")
    return ap


def main(
    argv: list[str] | None = None,
    *,
    load_generator: Callable[..., Any] = default_load_generator,
    load_embedder: Callable[..., Any] = default_load_embedder,
) -> dict[str, Any]:
    ap = build_parser()
    args = ap.parse_args(argv)
    records, input_info = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
        input_info["limit"] = args.limit
    embedder = load_embedder(args.embedding_model_id, args.embedding_revision, args.device)

    if args.baseline == "mem0":
        verify_structure()  # the vendored upstream files must match their pins before anything runs
        params = GenerationParams(seed=args.seed, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
        generator = load_generator(args.model_id, args.revision, args.device, args.shard)
        run_config = build_mem0_run_config(
            records, generator=generator, params=params, embedder_info=embedder.run_info(), input_info=input_info
        )
        result = run_mem0_baseline(records, run_config, generator=generator, params=params, embed_texts=embedder.embed)
        out_dir = args.out / run_config["group_id"]
        paths = write_mem0_outputs(out_dir, result)
    else:
        run_config = build_flat_run_config(
            records, embedder_info=embedder.run_info(), input_info=input_info, k=args.k, seed=args.seed
        )
        result = run_flat_baseline(records, run_config, embed_texts=embedder.embed, k=args.k)
        out_dir = args.out / run_config["group_id"]
        paths = write_flat_outputs(out_dir, result)

    (out_dir / "invocation.json").write_text(
        json.dumps({"argv": sys.argv if argv is None else ["dkmem.baselines.cli", *argv],
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{args.baseline}: {run_config['group_id']}")
    print(json.dumps(result.summary, ensure_ascii=False))
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return {"baseline": args.baseline, "group_id": run_config["group_id"], "dir": str(out_dir), "summary": result.summary}


if __name__ == "__main__":
    main()
