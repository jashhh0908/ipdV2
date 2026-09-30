"""DK-Mem distinction gating (research_idea_context.md Sec 4(b)).

Replaces the plain similarity check

    merge(a, b) if sim(gloss_a, gloss_b) > tau

with a deterministic compatibility gate on top of it:

    merge(a, b) if sim(gloss_a, gloss_b) > tau AND compatible(distinction_a, distinction_b)

``compatible()`` is a pure function over two write-side distinction dicts
(``Extraction.distinction`` -- never the hand-authored gold
``ProbeItem.distinction_value_*`` labels, which exist only for evaluation)
and the set of features considered identity-discriminative. It has three
outcomes, reproducing the convention already agreed and tested against in
``tests/fixtures/synthetic_fixtures.py``'s ``dkmem_v1`` policy:

- **compatible** -- for every discriminative feature, the two sides agree or
  at least one side doesn't mark it. Merge is then decided by similarity
  alone: ``merge`` if ``sim >= tau``, else ``keep_both``.
- **incompatible** -- some discriminative feature is marked on both sides
  with different values (e.g. kinship "bua" vs "mausi"). Always
  ``keep_both``, regardless of ``sim``.
- **underdetermined** -- some discriminative feature is marked on exactly
  one side. Always ``link_unresolved``, regardless of ``sim`` -- entries are
  linked rather than merged or duplicated, per the design's explicit
  underdetermined branch (this is the branch the research doc calls out as
  the one "where reviewers will look": blocking every uncertain merge causes
  bloat, merging them causes corruption).

When more than one discriminative feature is present, incompatible outranks
underdetermined (the more severe finding wins); this precedence is a
necessary completion of the rule, not specified verbatim in the research
doc, and is documented here explicitly rather than left implicit.

Scope of this module (Task 19 only):

- ``compatible()``/``gate()`` are pure dictionary comparisons -- no model or
  embedding calls, matching the spec's "compatible(): a dictionary
  comparison. Free."
- ``sim``/``tau`` are accepted as supplied inputs. No similarity or
  candidate-pair generation is implemented here.
- Only ``merge``, ``keep_both``, and ``link_unresolved`` are ever produced.
  ``supersede`` (recognizing a same-entity *update*, e.g. a changed
  location) needs a rule this module does not implement.
- No retrieval, and no Tier-1 batch execution over the 173-pair evaluation
  set -- this module only exposes the per-pair primitives a later task would
  call to build that pipeline's outputs.

Reused, not reinvented: this module produces ``dkmem.memory.schema.
MergeEvent`` records using the existing ``DECISIONS``. It defines no new
schema. ``MergeEvent`` has no dedicated "compatibility" field, so
``build_merge_event`` encodes it as the leading word of ``reason``
(``"compatible: ..."`` / ``"incompatible: ..."`` / ``"underdetermined:
..."``) -- callers that need it as a clean, separate value (e.g. Team B's
``pairwise_eval.jsonl``, which wants ``compatibility`` as its own field)
should call ``gate()`` directly rather than parse ``reason``; ``gate()``
returns it as a plain string, not embedded in text.
"""

from __future__ import annotations

from typing import Mapping

from dkmem.memory.schema import MergeEvent

__all__ = [
    "COMPATIBILITY",
    "DEFAULT_DISCRIMINATIVE_FEATURES",
    "compatible",
    "gate",
    "build_merge_event",
]

COMPATIBILITY = ("compatible", "incompatible", "underdetermined")

# Mirrors tests/fixtures/synthetic_fixtures.py's DISCRIMINATIVE_FEATURES (the
# team's agreed set); duplicated here as a default because the fixture
# module is test code, not an importable production constant. Callers may
# override per policy/ablation.
DEFAULT_DISCRIMINATIVE_FEATURES = frozenset(
    {"kinship", "register", "classifier", "evidentiality", "politeness", "temporal_deixis"}
)


def _compatibility_detail(
    distinction_a: Mapping[str, str],
    distinction_b: Mapping[str, str],
    discriminative_features,
) -> tuple[str, str | None, str | None, str | None]:
    """``(compatibility, feature, value_a, value_b)``.

    ``feature``/``value_a``/``value_b`` identify the discriminative feature
    responsible for a non-"compatible" result (the first found, in sorted
    order, for a stable choice when several features conflict); all three
    are ``None`` when compatible.
    """
    incompatible = None
    underdetermined = None
    for feature in sorted(discriminative_features):
        a = distinction_a.get(feature)
        b = distinction_b.get(feature)
        if a is not None and b is not None:
            if a != b and incompatible is None:
                incompatible = (feature, a, b)
        elif (a is not None) != (b is not None):
            if underdetermined is None:
                underdetermined = (feature, a, b)

    if incompatible is not None:
        feature, a, b = incompatible
        return "incompatible", feature, a, b
    if underdetermined is not None:
        feature, a, b = underdetermined
        return "underdetermined", feature, a, b
    return "compatible", None, None, None


def compatible(
    distinction_a: Mapping[str, str],
    distinction_b: Mapping[str, str],
    discriminative_features=DEFAULT_DISCRIMINATIVE_FEATURES,
) -> str:
    """One of ``COMPATIBILITY``: whether two extracted distinction dicts
    permit merging, per the rule documented at the top of this module.

    Deterministic, no side effects. ``distinction_a``/``distinction_b`` must
    be ``Extraction.distinction`` values (or equivalent ``dict[str, str]``),
    never the gold ``ProbeItem.distinction_value_*`` labels.
    """
    label, _, _, _ = _compatibility_detail(distinction_a, distinction_b, discriminative_features)
    return label


def gate(
    distinction_a: Mapping[str, str],
    distinction_b: Mapping[str, str],
    sim: float,
    tau: float,
    discriminative_features=DEFAULT_DISCRIMINATIVE_FEATURES,
) -> tuple[str, str, str]:
    """``(compatibility, decision, reason)`` for one pair.

    ``decision`` is one of ``dkmem.memory.schema.DECISIONS``, restricted here
    to ``merge``/``keep_both``/``link_unresolved`` (``supersede`` is out of
    scope for this module -- see module docstring).
    """
    compatibility, feature, a, b = _compatibility_detail(
        distinction_a, distinction_b, discriminative_features
    )

    if compatibility == "incompatible":
        return compatibility, "keep_both", f"incompatible {feature} values: {a} vs {b}"

    if compatibility == "underdetermined":
        marked = a if a is not None else b
        return (
            compatibility,
            "link_unresolved",
            f"{feature} marked ({marked}) on one side, unmarked on the other",
        )

    decision = "merge" if sim >= tau else "keep_both"
    comparison = ">=" if sim >= tau else "<"
    return compatibility, decision, f"compatible: sim {sim:.3g} {comparison} tau {tau:.3g}"


def build_merge_event(
    pair_id: str,
    distinction_a: Mapping[str, str],
    distinction_b: Mapping[str, str],
    *,
    sim: float,
    tau: float,
    policy: str,
    backbone: str,
    seed: int,
    prompt_id: str,
    discriminative_features=DEFAULT_DISCRIMINATIVE_FEATURES,
) -> MergeEvent:
    """Gate one pair and build the resulting ``MergeEvent``.

    All non-distinction fields (``pair_id``, ``policy``, ``backbone``,
    ``seed``, ``prompt_id``) are passed straight through to ``MergeEvent``;
    this function adds no schema fields of its own.
    """
    _compatibility, decision, reason = gate(
        distinction_a, distinction_b, sim, tau, discriminative_features
    )
    return MergeEvent(
        pair_id=pair_id,
        policy=policy,
        tau=tau,
        sim=sim,
        decision=decision,
        reason=reason,
        backbone=backbone,
        seed=seed,
        prompt_id=prompt_id,
    )
