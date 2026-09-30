"""Tests for dkmem.memory.matcher (the deterministic lexicon matcher).

Run: python -m unittest discover -s tests
"""

import unittest
from pathlib import Path

from dkmem.memory.lexicon import Lexicon, LexiconEntry, load_lexicon
from dkmem.memory.matcher import Match, find_matches, match_distinctions

LEXICON_PATH = Path(__file__).parent.parent / "dkmem" / "memory" / "distinction_features.json"


def entry(**overrides):
    d = dict(
        id="e1",
        lang="hi",
        distinction_class="register",
        match_type="exact_word",
        surface_forms=["tu"],
        value="tu",
        priority=0,
        notes="",
    )
    d.update(overrides)
    d["surface_forms"] = tuple(d["surface_forms"])
    return LexiconEntry(**d)


class TestRealLexiconMatching(unittest.TestCase):
    """Uses the real, pilot-reviewed tests/../distinction_features.json (12
    Hindi kinship entries, hi register tu/tum/aap, Turkish -mIş
    evidentiality)."""

    @classmethod
    def setUpClass(cls):
        cls.lex = load_lexicon(LEXICON_PATH)

    def test_all_12_kinship_entries_match_in_both_scripts(self):
        kinship = self.lex.for_lang_class("hi", "kinship")
        self.assertEqual(len(kinship), 12)
        for e in kinship:
            self.assertEqual(len(e.surface_forms), 2)  # one Latin, one Devanagari
            for form in e.surface_forms:
                with self.subTest(id=e.id, form=form):
                    sentence = f"{form} ghar mein rehti hai."
                    self.assertEqual(match_distinctions(self.lex, sentence, "hi"),
                                     {"kinship": e.value})

    def test_register_tu_tum_aap(self):
        register = self.lex.for_lang_class("hi", "register")
        self.assertEqual({e.value for e in register}, {"tu", "tum", "aap"})
        for e in register:
            for form in e.surface_forms:
                with self.subTest(id=e.id, form=form):
                    sentence = f"{form} kal office aaoge."
                    self.assertEqual(match_distinctions(self.lex, sentence, "hi"),
                                     {"register": e.value})

    def test_exact_word_is_case_insensitive(self):
        self.assertEqual(match_distinctions(self.lex, "TU aaoge.", "hi"), {"register": "tu"})
        self.assertEqual(match_distinctions(self.lex, "CHACHA ghar mein hai.", "hi"),
                         {"kinship": "chacha"})

    def test_suffix_evidentiality_turkish(self):
        self.assertEqual(
            match_distinctions(self.lex, "Ali Ankara'ya gitmiş.", "tr"),
            {"evidentiality": "reported/hearsay"},
        )

    def test_suffix_allomorphs(self):
        # Derive test words directly from the entry's own surface_forms,
        # rather than retyping the Turkish suffixes by hand, so this can't
        # drift from the actual (dotted/dotless i) characters in the JSON.
        (mis_entry,) = self.lex.for_lang_class("tr", "evidentiality")
        self.assertEqual(set(mis_entry.surface_forms), {"mış", "miş", "muş", "müş"})
        for suffix in mis_entry.surface_forms:
            word = "git" + suffix  # "git-" (go): a genuine 3-letter verb stem
            with self.subTest(word=word):
                self.assertEqual(
                    match_distinctions(self.lex, word, "tr"),
                    {"evidentiality": "reported/hearsay"},
                )

    def test_no_match_returns_empty_dict(self):
        self.assertEqual(match_distinctions(self.lex, "This is plain English.", "en"), {})

    def test_lang_filters_out_entries_from_other_languages(self):
        # "tu" is a real Hindi entry, but asking for "tr" must not match it.
        self.assertEqual(match_distinctions(self.lex, "Tu bahut accha hai.", "tr"), {})
        # Likewise the Turkish suffix must not be checked against Hindi.
        self.assertEqual(match_distinctions(self.lex, "gitmiş", "hi"), {})

    def test_punctuation_is_trimmed_from_token_ends(self):
        self.assertEqual(match_distinctions(self.lex, "Tu,", "hi"), {"register": "tu"})

    def test_empty_utterance(self):
        self.assertEqual(find_matches(self.lex, "", "hi"), ())
        self.assertEqual(match_distinctions(self.lex, "   ", "hi"), {})

    def test_no_cross_contamination_between_kinship_entries(self):
        # A sentence naming one relative must not also surface a neighbor's
        # value (e.g. mama/mami, tai/tau, mausa/mausi are minimal pairs).
        for word, expected in (("mama", "mama"), ("mami", "mami"), ("tau", "tau"),
                               ("tai", "tai"), ("mausa", "mausa"), ("mausi", "mausi")):
            with self.subTest(word=word):
                self.assertEqual(
                    match_distinctions(self.lex, f"Mera {word} Delhi mein rehta hai.", "hi"),
                    {"kinship": expected},
                )


class TestFalsePositives(unittest.TestCase):
    """Unrelated words/utterances must not spuriously match the real lexicon."""

    @classmethod
    def setUpClass(cls):
        cls.lex = load_lexicon(LEXICON_PATH)

    def test_unrelated_hindi_sentence(self):
        self.assertEqual(
            match_distinctions(self.lex, "Mera ghar bahut bada hai.", "hi"), {}
        )

    def test_common_function_words_do_not_match(self):
        for word in ("hai", "mein", "kal", "office", "yeh", "woh", "main", "uska", "ghar"):
            with self.subTest(word=word):
                self.assertEqual(match_distinctions(self.lex, word, "hi"), {})

    def test_kinship_term_as_substring_of_a_longer_word_does_not_match(self):
        # "chachaji" (chacha + honorific "-ji") is one token, not equal to
        # "chacha": exact_word requires whole-token equality.
        self.assertEqual(match_distinctions(self.lex, "chachaji ghar mein hai.", "hi"), {})
        self.assertEqual(match_distinctions(self.lex, "mamaji", "hi"), {})

    def test_similar_looking_turkish_word_is_not_evidentiality(self):
        # "yarış" (race) ends in "-rış", not any of mış/miş/muş/müş.
        self.assertEqual(match_distinctions(self.lex, "Bu bir yarış.", "tr"), {})

    def test_turkish_direct_past_is_not_tagged_as_hearsay(self):
        # "geldi" (came): witnessed/direct past (-dI), not the -mIş evidential.
        self.assertEqual(match_distinctions(self.lex, "Ali Ankara'ya geldi.", "tr"), {})

    def test_unrelated_language_text_under_hi_or_tr(self):
        self.assertEqual(match_distinctions(self.lex, "This is plain English text.", "hi"), {})
        self.assertEqual(match_distinctions(self.lex, "This is plain English text.", "tr"), {})


class TestSuffixStemLengthGuard(unittest.TestCase):
    """The Turkish evidentiality suffix must not match a bare suffix, or a
    token that leaves too little material to be a genuine verb stem (see
    "Suffix stem-length guard" in matcher.py's docstring)."""

    @classmethod
    def setUpClass(cls):
        cls.lex = load_lexicon(LEXICON_PATH)

    def test_bare_suffix_alone_does_not_match(self):
        for suffix in ("mış", "miş", "muş", "müş"):
            with self.subTest(suffix=suffix):
                self.assertEqual(match_distinctions(self.lex, suffix, "tr"), {})

    def test_one_character_stem_does_not_match(self):
        # Contrived: no real Turkish verb root is a single character, so a
        # 1-character remainder cannot be genuine -mIş morphology.
        self.assertEqual(match_distinctions(self.lex, "amiş", "tr"), {})

    def test_two_character_stem_does_match(self):
        # "de-" (say) and "ol-" (be/become) are genuine 2-letter verb roots.
        self.assertEqual(match_distinctions(self.lex, "demiş", "tr"),
                         {"evidentiality": "reported/hearsay"})
        self.assertEqual(match_distinctions(self.lex, "olmuş", "tr"),
                         {"evidentiality": "reported/hearsay"})

    def test_three_character_stem_still_matches(self):
        self.assertEqual(match_distinctions(self.lex, "gitmiş", "tr"),
                         {"evidentiality": "reported/hearsay"})


class TestTokenize(unittest.TestCase):
    """Indirectly via find_matches, since _tokenize is private."""

    def test_trims_mixed_script_punctuation(self):
        lex = Lexicon((entry(match_type="exact_word", surface_forms=["hai"], value="v"),))
        for utt in ("hai.", "hai,", "hai।", "hai。", '"hai"', "(hai)"):
            with self.subTest(utt=utt):
                self.assertEqual(len(find_matches(lex, utt, "hi")), 1, utt)

    def test_apostrophe_inside_token_is_preserved(self):
        # Turkish "Ankara'ya" must remain one token (apostrophe not at an edge).
        lex = Lexicon((entry(match_type="suffix", surface_forms=["ya"], value="v"),))
        (hit,) = find_matches(lex, "Ankara'ya gitti.", "hi")
        self.assertEqual(hit.token, "Ankara'ya")

    def test_all_punctuation_fragment_dropped(self):
        # "--" trims to an empty string and must vanish as a token entirely,
        # not just fail to match: an exact_word entry for "--" itself must
        # never match (nothing to trim down to), while "hai" on either side
        # still produces its own two matches (proving "--" didn't merge with
        # or swallow a neighboring token).
        lex = Lexicon((
            entry(id="dash", match_type="exact_word", surface_forms=["--"], value="v"),
            entry(id="hai", match_type="exact_word", surface_forms=["hai"], value="v"),
        ))
        hits = find_matches(lex, "hai -- hai", "hi")
        self.assertEqual([h.entry.id for h in hits], ["hai", "hai"])


class TestMatchTypes(unittest.TestCase):
    def test_prefix(self):
        lex = Lexicon((entry(match_type="prefix", surface_forms=["chach"], value="v"),))
        (hit,) = find_matches(lex, "chachi rehti hai", "hi")
        self.assertEqual((hit.token, hit.span), ("chachi", "chach"))
        self.assertEqual(find_matches(lex, "mausi rehti hai", "hi"), ())

    def test_suffix(self):
        lex = Lexicon((entry(match_type="suffix", surface_forms=["mış"], value="v"),))
        (hit,) = find_matches(lex, "gitmış", "hi")
        self.assertEqual(hit.span, "mış")

    def test_regex_group_zero_is_span(self):
        lex = Lexicon((entry(match_type="regex", surface_forms=[r"\d+"], value="v"),))
        (hit,) = find_matches(lex, "abc123def", "hi")
        self.assertEqual(hit.span, "123")

    def test_regex_no_implicit_case_folding(self):
        lex = Lexicon((entry(match_type="regex", surface_forms=["ABC"], value="v"),))
        self.assertEqual(find_matches(lex, "abc", "hi"), ())
        (hit,) = find_matches(lex, "ABC", "hi")
        self.assertEqual(hit.span, "ABC")

    def test_regex_explicit_inline_case_flag_is_respected(self):
        lex = Lexicon((entry(match_type="regex", surface_forms=["(?i)abc"], value="v"),))
        self.assertEqual(len(find_matches(lex, "ABC", "hi")), 1)

    def test_regex_anchors_are_relative_to_each_token_not_the_whole_utterance(self):
        # "ab$" matches the *first* (non-final) token here -- only possible
        # if "$" anchors per-token, since the whole utterance "ab xy" does
        # not itself end in "ab".
        end_anchored = Lexicon((entry(match_type="regex", surface_forms=["ab$"], value="v"),))
        (hit,) = find_matches(end_anchored, "ab xy", "hi")
        self.assertEqual((hit.token, hit.token_index), ("ab", 0))

        # "^ab" matches the *second* token here -- only possible if "^"
        # anchors per-token, since the whole utterance "xy ab" does not
        # itself start with "ab".
        start_anchored = Lexicon((entry(match_type="regex", surface_forms=["^ab"], value="v"),))
        (hit,) = find_matches(start_anchored, "xy ab", "hi")
        self.assertEqual((hit.token, hit.token_index), ("ab", 1))

    def test_entry_with_multiple_surface_forms_matches_any(self):
        lex = Lexicon((entry(match_type="exact_word", surface_forms=["aap", "आप"], value="aap"),))
        self.assertEqual(len(find_matches(lex, "aap", "hi")), 1)
        self.assertEqual(len(find_matches(lex, "आप", "hi")), 1)


class TestConflictResolution(unittest.TestCase):
    """Synthetic Lexicons built directly (bypassing load_lexicon's own
    conflict checks) so the matcher's tie-break logic can be tested in
    isolation, including inputs load_lexicon would itself reject."""

    def test_agreement_is_not_a_conflict(self):
        lex = Lexicon((
            entry(id="a", surface_forms=["tu"], value="tu"),
            entry(id="b", match_type="prefix", surface_forms=["t"], value="tu"),
        ))
        self.assertEqual(match_distinctions(lex, "tu", "hi"), {"register": "tu"})
        self.assertEqual(len(find_matches(lex, "tu", "hi")), 2)

    def test_different_classes_do_not_conflict(self):
        lex = Lexicon((
            entry(id="a", distinction_class="register", surface_forms=["tu"], value="tu"),
            entry(id="b", distinction_class="kinship", match_type="prefix",
                 surface_forms=["t"], value="someone"),
        ))
        self.assertEqual(match_distinctions(lex, "tu", "hi"),
                         {"register": "tu", "kinship": "someone"})

    def test_higher_priority_wins_regardless_of_position(self):
        lex = Lexicon((
            entry(id="early", surface_forms=["tu"], value="informal", priority=0),
            entry(id="late", distinction_class="register", match_type="exact_word",
                 surface_forms=["aap"], value="formal", priority=5),
        ))
        self.assertEqual(match_distinctions(lex, "tu ... aap", "hi"), {"register": "formal"})

    def test_earlier_token_wins_when_priority_ties(self):
        # Whichever entry matches the earlier token position wins -- not
        # whichever entry is listed first in the lexicon.
        lex = Lexicon((
            entry(id="informal-entry", surface_forms=["tu"], value="informal"),
            entry(id="formal-entry", surface_forms=["aap"], value="formal"),
        ))
        self.assertEqual(match_distinctions(lex, "tu kal aap", "hi"), {"register": "informal"})
        self.assertEqual(match_distinctions(lex, "aap kal tu", "hi"), {"register": "formal"})

    def test_longer_span_wins_when_priority_and_position_tie(self):
        # Both match the same token "chachi": exact_word matches the whole
        # word (span len 6), prefix matches only "chach" (span len 5).
        lex = Lexicon((
            entry(id="prefix-match", match_type="prefix", surface_forms=["chach"], value="short"),
            entry(id="exact-match", match_type="exact_word", surface_forms=["chachi"], value="long"),
        ))
        self.assertEqual(match_distinctions(lex, "chachi", "hi"), {"register": "long"})

    def test_lower_id_wins_as_final_tie_break(self):
        # Identical priority, token, and span length: only the id differs.
        # load_lexicon would reject this pair as a conflicting duplicate
        # mapping; Lexicon itself does not enforce that, so this exercises
        # the matcher's own final tie-break in isolation.
        lex = Lexicon((
            entry(id="z-entry", surface_forms=["tu"], value="Z"),
            entry(id="a-entry", surface_forms=["tu"], value="A"),
        ))
        self.assertEqual(match_distinctions(lex, "tu", "hi"), {"register": "A"})

    def test_tie_break_is_independent_of_entry_order(self):
        lex_a = Lexicon((
            entry(id="a", surface_forms=["tu"], value="informal", priority=5),
            entry(id="b", surface_forms=["aap"], value="formal", priority=0),
        ))
        lex_b = Lexicon((
            entry(id="b", surface_forms=["aap"], value="formal", priority=0),
            entry(id="a", surface_forms=["tu"], value="informal", priority=5),
        ))
        utt = "tu kal aap"
        self.assertEqual(match_distinctions(lex_a, utt, "hi"), match_distinctions(lex_b, utt, "hi"))


class TestFindMatchesShape(unittest.TestCase):
    def test_match_fields(self):
        lex = Lexicon((entry(),))
        (hit,) = find_matches(lex, "tu kal", "hi")
        self.assertIsInstance(hit, Match)
        self.assertEqual(hit.entry.id, "e1")
        self.assertEqual(hit.token_index, 0)
        self.assertEqual(hit.token, "tu")
        self.assertEqual(hit.span, "tu")

    def test_unknown_lang_yields_no_hits_not_an_error(self):
        lex = Lexicon((entry(),))
        self.assertEqual(find_matches(lex, "tu", "xx"), ())


if __name__ == "__main__":
    unittest.main()
