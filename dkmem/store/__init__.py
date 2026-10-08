"""Minimal in-memory DK-Mem memory store and consolidation flow (research code, no database).

``MemoryStore`` holds entries, unresolved links and an event log; ``consolidate`` writes one
entry through retrieve -> host proposal -> DK-Mem gate -> single-target merge / add / link.
The frozen A-D pipeline (``dkmem.pipeline``) is untouched; this package reuses its extraction,
distinction, gate, judge, similarity and schema code.
"""
