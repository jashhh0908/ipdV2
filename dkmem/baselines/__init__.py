"""Baselines of DKMEM_NEW_RESEARCH_IDEA.md Sec 6.3 that are not part of the A-D stage-attribution harness.

* ``dkmem.baselines.mem0`` -- a hash-pinned reproduction of the Mem0 OSS v1.0.11 add/update flow.
* ``dkmem.baselines.flat`` -- never-merge / flat dense RAG (store every utterance, never consolidate).

The prompt-informed merge judge is ``dkmem.memory.judge.MERGE_JUDGE_INFORMED_V1`` (a prompt, run through the
A-D pipeline with ``--judge-prompt``), and raise-tau is the offline sweep in ``dkmem.eval.sweep``. A-Mem and
LightMem are dropped baselines and are not implemented.
"""
