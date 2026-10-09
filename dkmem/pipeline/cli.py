"""Command line for the stage-attribution harness (Configs A-D; needs a GPU for real runs).

First real run (Qwen2.5-3B-Instruct, all 173 sanctioned Tier 1 records, seed 0)::

    python -m dkmem.pipeline.cli --configs A B C D --modes off lexicon lexicon+llm \\
        --model-id Qwen/Qwen2.5-3B-Instruct --seed 0 --tau-d 0.80 0.85 0.90 --out runs/

Writes ``<out>/<group_id>/`` per configuration (and, for Config D, per tau); see
``dkmem.pipeline.runner.write_outputs``. With no ``--input`` the sanctioned Tier 1
file is used (hash-checked, 173 records). Any other ``--input`` (e.g. the week-1 pilot
pairs) must be a JSONL of the same 8-field records; its sha256 is recorded and the run
config marks it ``sanctioned: false``.

Config D needs ``--tau-d`` (one or more values, no default: the right bge-m3 cosine
cutoff is an experimental result, not a constant); every value gets its own group
and its own pairwise_eval files, sharing one embedding pass. Configs A-C take no
threshold (their merge decision is an LLM judge) and their rows carry
``threshold: null``. Use ``--shard`` for a model that does not fit one T4 in fp16
(Qwen2.5-7B-Instruct): the weights are split over both GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from dkmem.backends.llm import GenerationParams
from dkmem.config import DKMEM_MODES, PIPELINE_CONFIG_IDS, get_pipeline_config
from dkmem.memory.judge import JUDGE_PROMPTS, MERGE_JUDGE_V1
from dkmem.memory.lexicon import load_lexicon
from dkmem.pipeline.runner import build_run_config, run_stage_attribution, write_outputs
from dkmem.pipeline.trace import sha256_file
from dkmem.tier1.io import Tier1InputError, Tier1Record, load_tier1_input

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "tier1" / "eval_input" / "teamA_tier1_v2.jsonl"
DEFAULT_LEXICON = ROOT / "memory" / "distinction_features.json"
EMBEDDING_METRIC_NAME = "bge_m3_dense_cosine_v1"


def load_records(path: Path) -> tuple[list[Tier1Record], dict]:
    """Records plus the ``input`` info recorded in the run config."""
    info = {"source_path": str(path), "file_sha256": sha256_file(path)}
    if path.resolve() == DEFAULT_INPUT.resolve():
        return load_tier1_input(path), {**info, "sanctioned": True}
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(Tier1Record.from_dict(json.loads(line)))
            except (json.JSONDecodeError, Tier1InputError) as e:
                raise SystemExit(f"{path}:{i}: {e}") from e
    return records, {**info, "sanctioned": False}


def default_load_generator(model_id: str, revision: str | None, device: str, shard: bool) -> Any:
    from dkmem.backends.llm import HFGenerator, ModelConfig

    cfg = ModelConfig(model_id=model_id, revision=revision, device=device)
    return HFGenerator.load_sharded(cfg) if shard else HFGenerator.load(cfg)


def default_load_embedder(model_id: str, revision: str | None, device: str) -> Any:
    from dkmem.backends.embedding import BgeM3Embedder, EmbeddingConfig

    return BgeM3Embedder.load(EmbeddingConfig(model_id=model_id, revision=revision, device=device))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=list(PIPELINE_CONFIG_IDS), choices=PIPELINE_CONFIG_IDS)
    ap.add_argument("--modes", nargs="+", default=list(DKMEM_MODES), choices=DKMEM_MODES)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", action="store_true", help="split the LLM over the visible GPUs (Qwen2.5-7B)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--tau-d", type=float, nargs="+", default=None,
                    help="merge threshold(s) of Config D: merge when cosine > tau (required for D)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N records (smoke test)")
    ap.add_argument("--embedding-model-id", default="BAAI/bge-m3")
    ap.add_argument("--embedding-revision", default=None)
    ap.add_argument("--judge-prompt", default=MERGE_JUDGE_V1.prompt_id, choices=sorted(JUDGE_PROMPTS),
                    help="merge judge of Configs A-C; merge_judge_informed_v1 is the prompt-informed-judge "
                         "baseline (use with --modes off). Config D has no judge and ignores it.")
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
            run_config = build_run_config(
                config_id, args.modes, records, generator=generator, lexicon_path=args.lexicon,
                params=params, similarity=metric if is_d else None, tau=tau,
                embedder_info=embedder.run_info() if is_d else None, input_info=input_info,
                judge_prompt=MERGE_JUDGE_V1 if is_d else JUDGE_PROMPTS[args.judge_prompt],
            )
            result = run_stage_attribution(
                records, config_id, lexicon, run_config, generator=generator, params=params,
                similarity=metric if is_d else None, tau=tau,
                judge_prompt=MERGE_JUDGE_V1 if is_d else JUDGE_PROMPTS[args.judge_prompt],
            )
            out_dir = args.out / run_config["group_id"]
            paths = write_outputs(out_dir, result)
            (out_dir / "invocation.json").write_text(
                json.dumps({"argv": sys.argv if argv is None else ["dkmem.pipeline.cli", *argv],
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
