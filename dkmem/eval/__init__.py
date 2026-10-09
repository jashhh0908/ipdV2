"""Offline evaluation of run groups: FCR / MCR, the raise-tau sweep, the DK-Mem ablation and a fairness check.

Everything here reads files a run already wrote (``pairwise_eval.jsonl``, ``trace.jsonl``, ``run_config.json``) plus
the gold relation of each Tier 1 pair, which Team B holds (``dkmem.eval.gold``): no model is run. The metric
definitions are those of ``dkmem/tier1_eval_contract.md`` (Sec 4-6), implemented once in ``dkmem.eval.metrics`` and
used for every system, so FCR and MCR mean the same thing for Configs A-D, the Mem0 reproduction and flat RAG.
"""
