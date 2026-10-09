"""Mem0 OSS v1.0.11 (tag ``v1.0.11``, commit ``144627c4ce5bc4db6acac17cbd158065f2b27a8d``) as a baseline.

The upstream package is not imported. Its add/update flow is re-implemented line by line in
``dkmem.baselines.mem0.pipeline`` and its prompts and helper functions are loaded from vendored,
hash-pinned copies of the upstream files (``dkmem.baselines.mem0.source``). Substitutions: the LLM is
the run's Qwen generator, the embedder is bge-m3, the vector store and history are in-memory.
"""
