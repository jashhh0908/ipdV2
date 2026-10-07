"""Which distinction classes are in scope (DKMEM_NEW_RESEARCH_IDEA.md Sec 6.1, 10, 11).

Single source of truth for the extraction prompt's key set, the gate's default
discriminative features, and the Tier 1 external class mapping.

- ``IN_SCOPE_CLASSES``: kinship (primary), register (honorific/register) and
  name_variant (secondary).
- ``DISCRIMINATIVE_CLASSES``: the subset the gate treats as cannot-link
  evidence. ``name_variant`` is recorded but deliberately not gating: Sec 6.1
  includes same-entity name-variant pairs that *should* merge (Priya /
  प्रिया), so a differing name variant must not block a merge. Whether and how
  name variants should constrain merging is not decided yet.
- ``OUT_OF_SCOPE_CLASSES``: dropped from the core plan (Sec 10/11
  "Removed"/"Out of scope"). Politeness (speech level) is not listed as in
  scope anywhere in Sec 6.1/11, so it is treated as dropped as well.
  ``LEGACY_CLASSES`` / ``LEGACY_DISCRIMINATIVE_CLASSES`` exist only so the
  frozen prompts ``mem0_extraction_v1``-``v3`` (which describe the old
  7-class scope) keep parsing and the old gate behaviour stays reproducible.
"""

from __future__ import annotations

__all__ = [
    "IN_SCOPE_CLASSES",
    "DISCRIMINATIVE_CLASSES",
    "OUT_OF_SCOPE_CLASSES",
    "LEGACY_CLASSES",
    "LEGACY_DISCRIMINATIVE_CLASSES",
]

IN_SCOPE_CLASSES = frozenset({"kinship", "register", "name_variant"})

DISCRIMINATIVE_CLASSES = frozenset({"kinship", "register"})

OUT_OF_SCOPE_CLASSES = frozenset({"politeness", "evidentiality", "classifier", "temporal_deixis"})

LEGACY_CLASSES = IN_SCOPE_CLASSES | OUT_OF_SCOPE_CLASSES

LEGACY_DISCRIMINATIVE_CLASSES = DISCRIMINATIVE_CLASSES | OUT_OF_SCOPE_CLASSES
