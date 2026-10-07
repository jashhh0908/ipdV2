"""Tests for dkmem.memory.lexicon (the distinction-lexicon schema/loader).

Run: python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path

from dkmem.memory.lexicon import (
    LEXICON_SCHEMA_VERSION,
    LexiconEntry,
    LexiconError,
    load_lexicon,
)
from dkmem.memory.scope import IN_SCOPE_CLASSES

LEXICON_PATH = Path(__file__).parent.parent / "dkmem" / "memory" / "distinction_features.json"


def entry(**overrides):
    d = dict(
        id="e1",
        lang="hi",
        distinction_class="register",
        match_type="exact_word",
        surface_forms=["tu"],
        value="tu",
    )
    d.update(overrides)
    return d


def wrap(*entries):
    return {"schema_version": LEXICON_SCHEMA_VERSION, "entries": list(entries)}


def write_tmp(data):
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp, ensure_ascii=False)
    tmp.close()
    return tmp.name


class TestRealLexiconFile(unittest.TestCase):
    """dkmem/memory/distinction_features.json (the pilot-reviewed lexicon,
    v1: 12 Hindi kinship terms and hi register tu/tum/aap) must load cleanly
    and have exactly this shape."""

    KINSHIP_VALUES = {
        "tau", "chacha", "tai", "chachi", "mama", "mausa",
        "mami", "mausi", "nana", "dadi", "nani", "bua",
    }
    REGISTER_VALUES = {"tu", "tum", "aap"}

    def test_loads(self):
        lex = load_lexicon(LEXICON_PATH)
        self.assertEqual(len(lex), 15)

    def test_classes_and_langs(self):
        lex = load_lexicon(LEXICON_PATH)
        self.assertEqual(lex.distinction_classes, {"kinship", "register"})
        self.assertEqual(lex.langs, {"hi"})

    def test_only_in_scope_classes(self):
        # The loader accepts any class; this keeps the shipped file within the
        # current scope (dkmem.memory.scope).
        lex = load_lexicon(LEXICON_PATH)
        self.assertLessEqual(lex.distinction_classes, IN_SCOPE_CLASSES)

    def test_kinship_entries(self):
        lex = load_lexicon(LEXICON_PATH)
        kinship = lex.for_lang_class("hi", "kinship")
        self.assertEqual(len(kinship), 12)
        self.assertEqual({e.value for e in kinship}, self.KINSHIP_VALUES)
        self.assertTrue(all(e.match_type == "exact_word" for e in kinship))

    def test_register_entries(self):
        lex = load_lexicon(LEXICON_PATH)
        register = lex.for_lang_class("hi", "register")
        self.assertEqual(len(register), 3)
        self.assertEqual({e.value for e in register}, self.REGISTER_VALUES)

    def test_no_unrelated_language_or_class(self):
        lex = load_lexicon(LEXICON_PATH)
        self.assertEqual(lex.for_lang_class("fr", "register"), ())
        self.assertEqual(lex.for_lang_class("ja", "classifier"), ())
        # Evidentiality (the former Turkish -mIş entry) is out of scope.
        self.assertEqual(lex.for_lang_class("tr", "evidentiality"), ())

    def test_every_hi_entry_has_latin_and_devanagari_forms(self):
        lex = load_lexicon(LEXICON_PATH)
        for e in lex:
            if e.lang != "hi":
                continue
            with self.subTest(id=e.id):
                self.assertEqual(len(e.surface_forms), 2)
                scripts = [any(0x0900 <= ord(c) <= 0x097F for c in f) for f in e.surface_forms]
                self.assertEqual(sorted(scripts), [False, True], "expected one Latin, one Devanagari form")

    def test_all_entries_documented(self):
        lex = load_lexicon(LEXICON_PATH)
        for e in lex:
            with self.subTest(id=e.id):
                self.assertTrue(e.notes.strip(), "entry has no documentation in notes")


class TestEntryValidation(unittest.TestCase):
    def assertBad(self, data):
        with self.assertRaises(LexiconError):
            LexiconEntry.from_dict(data)

    def test_valid_entry(self):
        e = LexiconEntry.from_dict(entry())
        self.assertEqual(e.surface_forms, ("tu",))
        self.assertEqual(e.priority, 0)
        self.assertEqual(e.notes, "")

    def test_missing_and_unknown_keys(self):
        d = entry()
        del d["value"]
        self.assertBad(d)
        self.assertBad({**entry(), "extra_field": 1})

    def test_bad_lang(self):
        for lang in ("HI", "hindi", "h", "", "hi-en", "hi1"):
            self.assertBad(entry(lang=lang))

    def test_bad_distinction_class(self):
        for cls in ("Kinship", "kin ship", "1kinship", "", "kin-ship"):
            self.assertBad(entry(distinction_class=cls))

    def test_bad_match_type(self):
        self.assertBad(entry(match_type="fuzzy"))

    def test_surface_forms_must_be_nonempty_list_of_strings(self):
        self.assertBad(entry(surface_forms=[]))
        self.assertBad(entry(surface_forms="tu"))
        self.assertBad(entry(surface_forms=["tu", ""]))
        self.assertBad(entry(surface_forms=["tu", 5]))

    def test_surface_forms_no_duplicates(self):
        self.assertBad(entry(surface_forms=["tu", "tu"]))

    def test_invalid_regex_rejected(self):
        self.assertBad(entry(match_type="regex", surface_forms=["("], value="x"))

    def test_valid_regex_accepted(self):
        e = LexiconEntry.from_dict(entry(match_type="regex", surface_forms=["ab+$"], value="x"))
        self.assertEqual(e.match_type, "regex")

    def test_empty_value_rejected(self):
        self.assertBad(entry(value=""))
        self.assertBad(entry(value="   "))

    def test_closed_class_value_enforced(self):
        self.assertBad(entry(distinction_class="politeness", value="polite"))
        ok = LexiconEntry.from_dict(entry(distinction_class="politeness", value="formal"))
        self.assertEqual(ok.value, "formal")

        self.assertBad(entry(distinction_class="evidentiality", value="heard"))
        LexiconEntry.from_dict(entry(distinction_class="evidentiality", value="reported/hearsay"))

        self.assertBad(entry(distinction_class="temporal_deixis", value="today"))
        LexiconEntry.from_dict(entry(distinction_class="temporal_deixis", value="tomorrow"))

    def test_open_class_accepts_free_value(self):
        e = LexiconEntry.from_dict(entry(distinction_class="kinship", value="mama"))
        self.assertEqual(e.value, "mama")

    def test_priority_must_be_int(self):
        self.assertBad(entry(priority="1"))
        self.assertBad(entry(priority=True))
        e = LexiconEntry.from_dict(entry(priority=5))
        self.assertEqual(e.priority, 5)

    def test_to_dict_roundtrip(self):
        e = LexiconEntry.from_dict(entry(surface_forms=["tu", "tum"]))
        d = e.to_dict()
        self.assertEqual(d["surface_forms"], ["tu", "tum"])
        self.assertEqual(LexiconEntry.from_dict(d), e)


class TestFileValidation(unittest.TestCase):
    def load(self, data):
        path = write_tmp(data)
        return load_lexicon(path)

    def assertFileBad(self, data, pattern=None):
        path = write_tmp(data)
        if pattern:
            with self.assertRaisesRegex(LexiconError, pattern):
                load_lexicon(path)
        else:
            with self.assertRaises(LexiconError):
                load_lexicon(path)

    def test_valid_minimal_file(self):
        lex = self.load(wrap(entry()))
        self.assertEqual(len(lex), 1)

    def test_missing_file(self):
        with self.assertRaises(LexiconError):
            load_lexicon("does/not/exist.json")

    def test_invalid_json(self):
        path = write_tmp({})  # placeholder path
        Path(path).write_text("{not valid json", encoding="utf-8")
        with self.assertRaisesRegex(LexiconError, "invalid JSON"):
            load_lexicon(path)

    def test_wrong_schema_version(self):
        self.assertFileBad({"schema_version": "2.0", "entries": [entry()]}, "schema_version")

    def test_missing_top_level_keys(self):
        self.assertFileBad({"entries": [entry()]})
        self.assertFileBad({"schema_version": LEXICON_SCHEMA_VERSION})

    def test_unknown_top_level_key(self):
        self.assertFileBad({**wrap(entry()), "extra": 1})

    def test_entries_must_be_nonempty_list(self):
        self.assertFileBad({"schema_version": LEXICON_SCHEMA_VERSION, "entries": []})
        self.assertFileBad({"schema_version": LEXICON_SCHEMA_VERSION, "entries": "not a list"})

    def test_bad_entry_reports_index(self):
        bad = entry()
        del bad["value"]
        self.assertFileBad(wrap(entry(), bad), r"entries\[1\]")

    def test_duplicate_id_rejected(self):
        self.assertFileBad(wrap(entry(id="dup"), entry(id="dup", surface_forms=["aap"])), "duplicate id")

    def test_conflicting_surface_form_rejected(self):
        self.assertFileBad(
            wrap(
                entry(id="a", surface_forms=["mama"], value="mama"),
                entry(id="b", surface_forms=["mama"], value="uncle"),
            ),
            "already maps to",
        )

    def test_identical_duplicate_mapping_accepted(self):
        lex = self.load(
            wrap(
                entry(id="a", surface_forms=["tu"], value="tu"),
                entry(id="b", surface_forms=["tu"], value="tu"),
            )
        )
        self.assertEqual(len(lex), 2)

    def test_same_surface_form_different_lang_or_class_is_not_a_conflict(self):
        lex = self.load(
            wrap(
                entry(id="a", lang="hi", value="tu"),
                entry(id="b", lang="ur", value="tu"),
                entry(id="c", distinction_class="kinship", surface_forms=["tu"], value="son"),
            )
        )
        self.assertEqual(len(lex), 3)

    def test_description_is_optional(self):
        data = wrap(entry())
        data["description"] = "hello"
        lex = self.load(data)
        self.assertEqual(len(lex), 1)


if __name__ == "__main__":
    unittest.main()
