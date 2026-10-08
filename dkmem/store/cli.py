"""Command line for the store runner (Configs A-D through the MemoryStore; needs a GPU for real runs).

Same flags, loaders and input handling as ``python -m dkmem.pipeline.cli`` (which it reuses), plus
``--k`` (retrieval depth) and ``--extraction-cache`` (an ``ExtractionCache`` JSONL for the V4 extractions).
For the Qwen2.5-3B run::

    python -m dkmem.store.cli --configs A B C D --modes off lexicon lexicon+llm \\
        --model-id Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \\
        --embedding-revision 5617a9f61b028005a4858fdac845db406aefb181 \\
        --seed 0 --tau-d 0.80 0.85 0.90 --out runs/

Writes ``<out>/store-<config>-...-<fingerprint>/`` per configuration (and per tau for D); see
``dkmem.store.runner``. The A-D harness's own outputs are untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from dkmem.backends.llm import GenerationParams
from dkmem.config import get_pipeline_config
from dkmem.memory.cache import ExtractionCache
from dkmem.memory.lexicon import load_lexicon
from dkmem.pipeline.cli import (
    EMBEDDING_METRIC_NAME,
    build_parser as build_pipeline_parser,
    default_load_embedder,
    default_load_generator,
    load_records,
)
from dkmem.store.runner import (
    DEFAULT_K,
    build_store_run_config,
    run_store_attribution,
    write_store_outputs,
)

__all__ = ["build_parser", "main"]


def build_parser():
    ap = build_pipeline_parser()
    ap.description = __doc__
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="retrieval depth: candidates compared per write")
    ap.add_argument("--extraction-cache", type=Path, default=None,
                    help="ExtractionCache JSONL for the V4 extractions (read and appended)")
    return ap


def main(
    argv: list[str] | None = None,
    *,
    load_generator: Callable[..., Any] = default_load_generator,
    load_embedder: Callable[..., Any] = default_load_embedder,
) -> list[dict[str, Any]]:
    """Run the requested configurations; returns one ``{config, tau, group_id, dir, summary}`` per run."""
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.k < 1:
        ap.error("--k must be >= 1")
    if "D" in args.configs:
        if not args.tau_d:
            ap.error("Config D requires --tau-d")
        if any(not 0.0 <= t <= 1.0 for t in args.tau_d):
            ap.error("--tau-d values must be in [0, 1]")
    elif args.tau_d:
        ap.error("--tau-d only applies to Config D")

    records, input_info = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
        input_info["limit"] = args.limit
    lexicon = load_lexicon(args.lexicon)

    params = GenerationParams(seed=args.seed, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    needs_llm = any(
        get_pipeline_config(c).merge_mechanism == "llm_judge" for c in args.configs
    ) or "lexicon+llm" in args.modes
    generator = load_generator(args.model_id, args.revision, args.device, args.shard) if needs_llm else None
    cache = ExtractionCache(args.extraction_cache) if args.extraction_cache else None

    metric = embedder = None
    if "D" in args.configs:
        from dkmem.backends.embedding import cosine_metric

        embedder = load_embedder(args.embedding_model_id, args.embedding_revision, args.device)
        metric = cosine_metric(embedder.embed, EMBEDDING_METRIC_NAME)

    runs = []
    for config_id in args.configs:
        taus = args.tau_d if config_id == "D" else [None]
        for tau in taus:
            is_d = config_id == "D"
            run_config = build_store_run_config(
                config_id, args.modes, records, generator=generator, lexicon_path=args.lexicon,
                params=params, similarity=metric if is_d else None, tau=tau,
                embedder_info=embedder.run_info() if is_d else None, input_info=input_info, k=args.k,
            )
            result = run_store_attribution(
                records, config_id, lexicon, run_config, generator=generator, params=params,
                similarity=metric if is_d else None, tau=tau, k=args.k, extraction_cache=cache,
            )
            out_dir = args.out / run_config["group_id"]
            paths = write_store_outputs(out_dir, result)
            (out_dir / "invocation.json").write_text(
                json.dumps({"argv": sys.argv if argv is None else ["dkmem.store.cli", *argv],
                            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"config {config_id}" + (f" tau {tau}" if tau is not None else "") + f": {run_config['group_id']}")
            print(json.dumps(result.summary, ensure_ascii=False))
            for name, path in paths.items():
                print(f"  {name}: {path}")
            runs.append({"config": config_id, "tau": tau, "group_id": run_config["group_id"],
                         "dir": str(out_dir), "summary": result.summary})
    return runs


if __name__ == "__main__":
    main()
