"""Team B Tier 1 evaluation harness for the two in-scope DK-Mem strategies:
``dk-mem-lexicon`` (no LLM calls) and ``dk-mem-lexicon-llm`` (lexicon +
Mem0-style baseline). See ``dkmem/tier1/runner.py`` for the entry points.

This package is glue and translation on top of the unmodified core library
(``dkmem.memory.extract``, ``.lexicon``, ``.matcher``, ``.gate``,
``.similarity``, ``.schema``); it defines Team B's external record shapes,
not a new consolidation policy.
"""
