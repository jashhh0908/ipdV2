"""Synthetic schema fixtures: 10 ProbeItems, their Extractions, and MergeEvents.

These are hand-written test fixtures, not empirical results. Regenerate the
JSONL files (written next to this file) from the repo root with:

    python -m tests.fixtures.synthetic_fixtures

Conventions:
- ``Extraction.distinction`` keys equal the probe's ``distinction_class``; a
  control probe uses ``distinction_class="control_<feature>"`` with key
  ``<feature>``.
- ``sim`` is the gloss similarity (synthetic values), except for
  ``surface_only`` events, whose ``sim`` is the similarity of the original
  surface utterances (``surface_similarity``).
- ``mem0_style`` and ``surface_only`` consolidate (merge/supersede) iff
  ``sim >= tau``; ``no_consolidate`` always keeps both.
- ``dkmem_v1``: only features in ``DISCRIMINATIVE_FEATURES`` gate merging.
  Distinct values -> keep_both; one side marked, the other unmarked ->
  link_unresolved; otherwise merge iff ``sim >= tau``, else keep_both.
  Name variants are not discriminative.
"""

import difflib
import json
from pathlib import Path

from dkmem.memory.schema import Extraction, MergeEvent, ProbeItem, write_jsonl


OUT_DIR = Path(__file__).parent

BACKBONE = "fixture-backbone"
PROMPT_ID = "mem0_extraction_v1"
SEED = 42
TAU = 0.85

# Team decision for the current synthetic fixtures. Fixture class names map to
# the agreed features: register = honorific/register, temporal_deixis =
# spatial_temporal_deixis. name_variant is deliberately not discriminative.
DISCRIMINATIVE_FEATURES = frozenset(
    {"kinship", "register", "classifier", "evidentiality", "politeness", "temporal_deixis"}
)


def surface_similarity(utt_a: str, utt_b: str) -> float:
    """Deterministic fixture stand-in for surface-utterance similarity.

    Character-sequence ratio (difflib), rounded to 2 decimals. Real runs use
    the experiment's embedding model; this only makes the fixtures reproducible.
    """
    return round(difflib.SequenceMatcher(None, utt_a, utt_b).ratio(), 2)


PROBES = [
    ProbeItem(
        pair_id="probe_001",
        utt_a="Meri chachi Mumbai mein rehti hai.",
        utt_b="Meri mausi Mumbai mein rehti hai.",
        lang="hi",
        distinction_class="kinship",
        distinction_value_a="chachi",
        distinction_value_b="mausi",
        gold_same_entity=False,
        retrieval_query="Meri chachi kahan rehti hai?",
        notes="Classic kinship conflation minimal pair; gloss collapses both to aunt.",
    ),
    ProbeItem(
        pair_id="probe_002",
        utt_a="Meri chachi Mumbai mein rehti hai.",
        utt_b="Meri chachi Pune mein shift ho gayi hai.",
        lang="hi",
        distinction_class="kinship",
        distinction_value_a="chachi",
        distinction_value_b="chachi",
        gold_same_entity=True,
        retrieval_query="Meri chachi ab kahan rehti hai?",
        notes="Same entity with updated location; exercises supersession/merge.",
    ),
    ProbeItem(
        pair_id="probe_003",
        utt_a="Tum kal office aaoge.",
        utt_b="Aap kal office aaoge.",
        lang="hi",
        distinction_class="register",
        distinction_value_a="tum",
        distinction_value_b="aap",
        gold_same_entity=True,
        retrieval_query="Kal office kaun aa raha hai?",
        notes="Register distinction; same referent but different politeness level.",
    ),
    ProbeItem(
        pair_id="probe_004",
        utt_a="Meri chachi Mumbai mein rehti hai.",
        utt_b="Meri aunty Mumbai mein rehti hai.",
        lang="hi-en",
        distinction_class="kinship",
        distinction_value_a="chachi",
        distinction_value_b=None,
        gold_same_entity=True,
        retrieval_query="Meri chachi kahan rehti hai?",
        notes=(
            "Underdetermined kinship: 'chachi' is marked, English 'aunty' is unmarked. "
            "Gold assumes the same aunt; the text alone does not settle it."
        ),
    ),
    ProbeItem(
        pair_id="probe_005",
        utt_a="Priya ne mera call uthaya.",
        utt_b="प्रिया ने मेरा call उठाया.",
        lang="hi-en",
        distinction_class="name_variant",
        distinction_value_a="Priya-latin",
        distinction_value_b="Priya-devanagari",
        gold_same_entity=True,
        retrieval_query="Who answered my call?",
        notes="Surface/script variant of the same person; should remain merge-compatible.",
    ),
    ProbeItem(
        pair_id="probe_006",
        utt_a="Ali Ankara'ya gitmiş.",
        utt_b="Ali Ankara'ya gitti.",
        lang="tr",
        distinction_class="evidentiality",
        distinction_value_a="reported/hearsay",
        distinction_value_b="direct/confirmed",
        gold_same_entity=True,
        retrieval_query="Ali Ankara'ya gitti mi?",
        notes="Same person/event with different evidential status; exercises non-identity of feature vs entity.",
    ),
    ProbeItem(
        pair_id="probe_007",
        utt_a="水を三杯飲みました。",
        utt_b="水を三本飲みました。",
        lang="ja",
        distinction_class="classifier",
        distinction_value_a="cup",
        distinction_value_b="long-object",
        gold_same_entity=False,
        retrieval_query="How many water servings did the user drink?",
        notes="Classifier minimal pair: 杯 (cups) vs 本 (bottles); gloss collapses both to servings.",
    ),
    ProbeItem(
        pair_id="probe_008",
        utt_a="선생님께 이메일을 보냈어.",
        utt_b="선생님께 이메일을 보냈습니다.",
        lang="ko",
        distinction_class="politeness",
        distinction_value_a="informal",
        distinction_value_b="formal",
        gold_same_entity=True,
        retrieval_query="Did the user email the teacher?",
        notes="Same event with different Korean speech levels.",
    ),
    ProbeItem(
        pair_id="probe_009",
        utt_a="मैं कल मुंबई गया था।",
        utt_b="मैं कल मुंबई जाऊँगा।",
        lang="hi",
        distinction_class="temporal_deixis",
        distinction_value_a="yesterday",
        distinction_value_b="tomorrow",
        gold_same_entity=False,
        retrieval_query="When did the user go to Mumbai?",
        notes="Hindi 'kal' is disambiguated by tense; tests temporal false consolidation.",
    ),
    ProbeItem(
        pair_id="probe_010",
        utt_a="Mein Bruder lebt in Berlin.",
        utt_b="Meine Schwester studiert in Hamburg.",
        lang="de",
        distinction_class="control_kinship",
        distinction_value_a="brother",
        distinction_value_b="sister",
        gold_same_entity=False,
        retrieval_query="Wo lebt der Bruder des Benutzers?",
        notes=(
            "Non-Indic negative control: English lexicalizes brother/sister, so this is "
            "not a conflation probe; unrelated facts that no policy should consolidate."
        ),
    ),
]


def make_extraction(probe: ProbeItem, side: str, *, gloss: str, distinction: dict[str, str],
                    surface: str, lang_profile: dict[str, float]) -> Extraction:
    return Extraction(
        pair_id=probe.pair_id,
        side=side,
        raw_output=json.dumps(
            {"gloss": gloss, "distinction": distinction, "surface": surface},
            ensure_ascii=False,
        ),
        gloss=gloss,
        distinction=distinction,
        surface=surface,
        lang_profile=lang_profile,
        backbone=BACKBONE,
        prompt_id=PROMPT_ID,
        seed=SEED,
    )


EXTRACTIONS = [
    make_extraction(PROBES[0], "a", gloss="user's aunt lives in Mumbai", distinction={"kinship": "chachi"}, surface=PROBES[0].utt_a, lang_profile={"hi": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[0], "b", gloss="user's aunt lives in Mumbai", distinction={"kinship": "mausi"}, surface=PROBES[0].utt_b, lang_profile={"hi": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[1], "a", gloss="user's aunt lives in Mumbai", distinction={"kinship": "chachi"}, surface=PROBES[1].utt_a, lang_profile={"hi": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[1], "b", gloss="user's aunt moved to Pune", distinction={"kinship": "chachi"}, surface=PROBES[1].utt_b, lang_profile={"hi": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[2], "a", gloss="addressee will come to the office tomorrow", distinction={"register": "tum"}, surface=PROBES[2].utt_a, lang_profile={"hi": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[2], "b", gloss="addressee will come to the office tomorrow", distinction={"register": "aap"}, surface=PROBES[2].utt_b, lang_profile={"hi": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[3], "a", gloss="user's aunt lives in Mumbai", distinction={"kinship": "chachi"}, surface=PROBES[3].utt_a, lang_profile={"hi": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[3], "b", gloss="user's aunt lives in Mumbai", distinction={}, surface=PROBES[3].utt_b, lang_profile={"hi": 0.83, "en": 0.17, "script_mix": 0.0}),
    make_extraction(PROBES[4], "a", gloss="Priya answered the user's call", distinction={"name_variant": "Priya-latin"}, surface=PROBES[4].utt_a, lang_profile={"hi": 0.8, "en": 0.2, "script_mix": 0.0}),
    make_extraction(PROBES[4], "b", gloss="Priya answered the user's call", distinction={"name_variant": "Priya-devanagari"}, surface=PROBES[4].utt_b, lang_profile={"hi": 0.8, "en": 0.2, "script_mix": 0.2}),
    make_extraction(PROBES[5], "a", gloss="Ali went to Ankara", distinction={"evidentiality": "reported/hearsay"}, surface=PROBES[5].utt_a, lang_profile={"tr": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[5], "b", gloss="Ali went to Ankara", distinction={"evidentiality": "direct/confirmed"}, surface=PROBES[5].utt_b, lang_profile={"tr": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[6], "a", gloss="user drank three servings of water", distinction={"classifier": "cup"}, surface=PROBES[6].utt_a, lang_profile={"ja": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[6], "b", gloss="user drank three servings of water", distinction={"classifier": "long-object"}, surface=PROBES[6].utt_b, lang_profile={"ja": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[7], "a", gloss="user emailed the teacher", distinction={"politeness": "informal"}, surface=PROBES[7].utt_a, lang_profile={"ko": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[7], "b", gloss="user emailed the teacher", distinction={"politeness": "formal"}, surface=PROBES[7].utt_b, lang_profile={"ko": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[8], "a", gloss="user went to Mumbai yesterday", distinction={"temporal_deixis": "yesterday"}, surface=PROBES[8].utt_a, lang_profile={"hi": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[8], "b", gloss="user will go to Mumbai tomorrow", distinction={"temporal_deixis": "tomorrow"}, surface=PROBES[8].utt_b, lang_profile={"hi": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[9], "a", gloss="user's brother lives in Berlin", distinction={"kinship": "brother"}, surface=PROBES[9].utt_a, lang_profile={"de": 1.0, "en": 0.0, "script_mix": 0.0}),
    make_extraction(PROBES[9], "b", gloss="user's sister studies in Hamburg", distinction={"kinship": "sister"}, surface=PROBES[9].utt_b, lang_profile={"de": 1.0, "en": 0.0, "script_mix": 0.0}),
]


# Synthetic expected outcomes. These are test fixtures, not empirical results.
MERGE_EVENTS = []
PROBES_BY_ID = {p.pair_id: p for p in PROBES}


def add_events(pair_id: str, sim: float, decisions: dict[str, tuple[str, str]]) -> None:
    """Add one MergeEvent per policy; ``sim`` is the gloss similarity.

    ``surface_only`` events get the surface-utterance similarity instead.
    """
    probe = PROBES_BY_ID[pair_id]
    for policy, (decision, reason) in decisions.items():
        event_sim = (
            surface_similarity(probe.utt_a, probe.utt_b) if policy == "surface_only" else sim
        )
        MERGE_EVENTS.append(
            MergeEvent(
                pair_id=pair_id,
                policy=policy,
                tau=TAU,
                sim=event_sim,
                decision=decision,
                reason=reason,
                backbone=BACKBONE,
                seed=SEED,
                prompt_id=PROMPT_ID,
            )
        )


SURFACE_ABOVE = ("merge", "surface similarity >= tau")
SURFACE_BELOW = ("keep_both", "surface similarity < tau")
NEVER = ("keep_both", "policy never consolidates")

add_events(
    "probe_001", 1.0,
    {
        "mem0_style": ("merge", "identical gloss; distinction ignored in baseline"),
        "surface_only": SURFACE_ABOVE,  # 0.90
        "no_consolidate": NEVER,
        "dkmem_v1": ("keep_both", "incompatible kinship values: chachi vs mausi"),
    },
)
add_events(
    "probe_002", 0.91,
    {
        "mem0_style": ("supersede", "same normalized entity with newer location"),
        "surface_only": SURFACE_BELOW,  # 0.73
        "no_consolidate": NEVER,
        "dkmem_v1": ("merge", "compatible kinship values; gloss similarity >= tau"),
    },
)
add_events(
    "probe_003", 0.95,
    {
        "mem0_style": ("merge", "high semantic similarity"),
        "surface_only": SURFACE_ABOVE,  # 0.86
        "no_consolidate": NEVER,
        "dkmem_v1": ("keep_both", "incompatible register values: tum vs aap"),
    },
)
add_events(
    "probe_004", 1.0,
    {
        "mem0_style": ("merge", "identical gloss; kinship specificity ignored in baseline"),
        "surface_only": SURFACE_ABOVE,  # 0.87
        "no_consolidate": NEVER,
        "dkmem_v1": ("link_unresolved", "kinship marked (chachi) on one side, unmarked on the other"),
    },
)
add_events(
    "probe_005", 0.98,
    {
        "mem0_style": ("merge", "same name and event after normalization"),
        "surface_only": SURFACE_BELOW,  # 0.35
        "no_consolidate": NEVER,
        "dkmem_v1": ("merge", "name variant is not discriminative; gloss similarity >= tau"),
    },
)
add_events(
    "probe_006", 0.97,
    {
        "mem0_style": ("merge", "same gloss"),
        "surface_only": SURFACE_ABOVE,  # 0.93
        "no_consolidate": NEVER,
        "dkmem_v1": ("keep_both", "incompatible evidentiality values: reported/hearsay vs direct/confirmed"),
    },
)
add_events(
    "probe_007", 1.0,
    {
        "mem0_style": ("merge", "identical gloss; classifier dropped in normalization"),
        "surface_only": SURFACE_ABOVE,  # 0.90
        "no_consolidate": NEVER,
        "dkmem_v1": ("keep_both", "incompatible classifier values: cup vs long-object"),
    },
)
add_events(
    "probe_008", 0.96,
    {
        "mem0_style": ("merge", "same normalized event"),
        "surface_only": SURFACE_ABOVE,  # 0.87
        "no_consolidate": NEVER,
        "dkmem_v1": ("keep_both", "incompatible politeness values: informal vs formal"),
    },
)
add_events(
    "probe_009", 0.92,
    {
        "mem0_style": ("merge", "shared lexical item may look similar after normalization"),
        "surface_only": SURFACE_BELOW,  # 0.80
        "no_consolidate": NEVER,
        "dkmem_v1": ("keep_both", "incompatible temporal deixis values: yesterday vs tomorrow"),
    },
)
add_events(
    "probe_010", 0.33,
    {
        "mem0_style": ("keep_both", "low similarity and distinct kinship"),
        "surface_only": SURFACE_BELOW,  # 0.51
        "no_consolidate": NEVER,
        "dkmem_v1": ("keep_both", "incompatible kinship values: brother vs sister"),
    },
)


def main(out_dir: Path = OUT_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "probe_items.jsonl", PROBES)
    write_jsonl(out_dir / "extractions.jsonl", EXTRACTIONS)
    write_jsonl(out_dir / "merge_events.jsonl", MERGE_EVENTS)
    print(f"Wrote {len(PROBES)} ProbeItems")
    print(f"Wrote {len(EXTRACTIONS)} Extractions")
    print(f"Wrote {len(MERGE_EVENTS)} MergeEvents")


if __name__ == "__main__":
    main()
