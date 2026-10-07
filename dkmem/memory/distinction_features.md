# Distinction lexicon format

`distinction_features.json` is a static, hand-curated table used by the
DK-Mem "lexicon-first" extractor described in `DKMEM_NEW_RESEARCH_IDEA.md`
Sec 5.3: a deterministic dictionary that maps a *surface marker* in a
source-language utterance (a word, prefix, suffix, or pattern) to the
`Extraction.distinction` key/value it implies, without calling a model.

This file is loaded and validated by `dkmem/memory/lexicon.py`
(`load_lexicon`). That loader is all that exists so far — it parses and
validates the JSON into `LexiconEntry`/`Lexicon` objects. It does **not**
scan an utterance for matches; that matcher is a separate, later piece of
work that will consume the `Lexicon` this file loads.

The current file (v1) is the pilot-reviewed core: 12 Hindi kinship terms and
Hindi register `tu`/`tum`/`aap` (15 entries). It follows the current scope
(`dkmem/memory/scope.py`: kinship, register, name_variant); the earlier Turkish
`-mIş` evidentiality entry was removed because evidentiality is out of scope.
The Kinbank-generated kinship slice and name-variant entries are still to come
— see the file's own `description` field.

## Top-level shape

```json
{
  "schema_version": "1.0",
  "description": "free-text, optional",
  "entries": [ { ... }, { ... } ]
}
```

- `schema_version` must be exactly `"1.0"` (the loader rejects anything
  else). If the entry shape below ever needs a breaking change, that change
  bumps `LEXICON_SCHEMA_VERSION` in `lexicon.py` in the same commit.
- `entries` must be a non-empty list. No other top-level keys are allowed.

## Entry fields

Each element of `entries` is one surface-marker -> value mapping:

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | yes | Unique across the whole file. Suggested convention: `<lang>-<distinction_class>-<short-slug>`, e.g. `"hi-kinship-mama"`. The loader rejects duplicates. |
| `lang` | string | yes | A single lowercase 2-3 letter language code (`"hi"`, `"tr"`, `"ja"`, `"ko"`, `"de"`, ...). One entry = one language; add a separate entry per language even if the marker is conceptually the same. |
| `distinction_class` | string | yes | Which `Extraction.distinction` key this fills. Must look like a snake_case identifier (`^[a-z][a-z0-9_]*$`). See "Distinction classes" below. |
| `match_type` | string | yes | One of `"exact_word"`, `"prefix"`, `"suffix"`, `"regex"`. See "Match types" below. This set is fixed by code, not data — see the note at the end. |
| `surface_forms` | list of strings | yes | One or more markers that all map to the same `value` (spelling/script variants, or suffix allomorphs). Must be non-empty, with no duplicate entries. For `match_type: "regex"`, each string is a regular expression (compiled at load time; an invalid pattern is rejected immediately). |
| `value` | string | yes | The value written into `Extraction.distinction[distinction_class]`. Free text in general, **except** for the three classes the extraction pipeline already treats as closed-vocabulary — see "Closed-vocabulary classes" below, where the loader enforces the exact allowed set. |
| `priority` | int | no (default `0`) | A hint for the future matcher to break ties when multiple entries could match overlapping text (e.g. prefer higher priority, or the longest surface form). Not used by anything in this module yet. |
| `notes` | string | no (default `""`) | Free-text documentation: gloss, examples, caveats. Put linguistic justification here, not in `id`. |

Any key not in this table, or a missing required key, makes the whole file
fail to load (with the offending entry's index reported).

## Match types

These describe the *mechanism* the matcher (`dkmem/memory/matcher.py`) uses,
not the linguistics. The set is intentionally small and fixed in code
(`MATCH_TYPES` in `lexicon.py`); adding a new one is a code change, because
something has to implement it, not just a data change.

The matcher splits an utterance on whitespace and trims leading/trailing
punctuation from each token (see `matcher.py`'s module docstring for exact
rules); every match type below is applied **per token**, not against the
whole utterance. A script without spaces between words (Japanese, Chinese)
is therefore not segmented into words — such text is effectively one long
token per whitespace-delimited chunk.

- **`exact_word`** — the marker equals a whole token in the utterance (e.g.
  a pronoun like Hindi `"tu"`).
- **`prefix`** — a token starts with the marker.
- **`suffix`** — a token ends with the marker. This is the usual choice for
  verb/case endings. The matcher additionally
  requires at least 2 characters of token before the suffix, so the bare
  suffix alone (or a 1-character remainder) never counts as a match — see
  "Suffix stem-length guard" in `matcher.py`'s docstring. This is a
  mechanical safeguard, not morphological validation: it cannot tell a
  genuine verbal use of a suffix from an unrelated grammatical use that
  happens to share the same ending (e.g. a Turkish `-mIş` participle that
  is not reported speech).
- **`regex`** — the marker is a regular expression, tested against a token
  with `re.search`. Use this when a simple prefix/suffix isn't enough (e.g.
  a counter word that follows a numeral). `^`/`$` anchor to the token's
  boundaries, not the whole utterance — anchor your pattern so the intent
  is unambiguous.

`exact_word`/`prefix`/`suffix` comparisons are case-insensitive (Unicode
`casefold`), so `"aap"` matches a capitalized `"Aap"` at the start of a
sentence; write surface forms in whatever casing is clearest to read.
`regex` patterns are matched as written, with **no** implicit
case-folding — add `(?i)` yourself if you want that.

## Distinction classes

`distinction_class` is deliberately **not** a closed enum in the loader, so
the format can hold any snake_case class. What the project actually uses is set
by `dkmem/memory/scope.py`:

- **In scope:** `kinship` (primary), `register` (honorific/register) and
  `name_variant` (secondary). Of these, `kinship` and `register` are the
  cannot-link classes the gate compares (`DISCRIMINATIVE_CLASSES`).
- **Out of scope** (dropped from the core plan): `politeness`,
  `evidentiality`, `classifier`, `temporal_deixis`. Do not add entries for them
  to the shipped file; `tests/test_lexicon.py` checks that the real file
  contains only in-scope classes.

`name_variant` is in scope but has **no entries yet**. A person's name is an
open vocabulary, so how a dictionary should cover it (and what a mismatch
between two variants of a name should do at merge time) is still undecided.

If you need a new class, tell the extraction-pipeline maintainer: the key sets
in `dkmem/memory/extract.py` (`DISTINCTION_KEYS`) and `dkmem/memory/scope.py`
must allow it before a prompt or the gate can use it.

## Closed-vocabulary classes

For `politeness`, `evidentiality`, and `temporal_deixis`, `extract.py`
restricts the model's output to a fixed value set (`CLOSED_DISTINCTION_VALUES`)
under the legacy prompts `mem0_extraction_v1`-`v3`. These classes are now out
of scope (the current default prompt, v4, does not allow them), but the
lexicon loader still enforces the *same* sets for any entry that uses them, so
a lexicon entry can never produce a value those prompts would reject:

| `distinction_class` | allowed `value`s |
|---|---|
| `politeness` | `"formal"`, `"informal"` |
| `evidentiality` | `"direct/confirmed"`, `"reported/hearsay"` |
| `temporal_deixis` | `"yesterday"`, `"tomorrow"` |

All other classes (`kinship`, `register`, and any new class you add) take
free-text values — see `dkmem/memory/prompts.py` for the value conventions the
extraction prompts already use (`kinship` values are the romanized source word,
or its English translation when English has an exact equivalent; `register`
values are the romanized second-person pronoun).

## Conflict and duplicate rules

The loader checks the whole file together, not just each entry in
isolation:

- **Duplicate `id`** across entries fails to load.
- **The same `(lang, distinction_class, match_type, surface_form)` mapping
  to two different `value`s** fails to load — this usually means two
  entries disagree about a marker and one of them is wrong.
- **The same mapping repeated with an identical `value`** (e.g. the same
  surface form added twice by accident, or split across two entries with
  different `notes`) is accepted; it's redundant but not a conflict.

## Worked examples

```json
{
  "id": "hi-kinship-mama",
  "lang": "hi",
  "distinction_class": "kinship",
  "match_type": "exact_word",
  "surface_forms": ["mama"],
  "value": "mama",
  "notes": "Maternal uncle (mother's brother). No single-word English equivalent, so the value is the romanized source term, matching prompts.py's kinship convention."
}
```

A `suffix` entry (format illustration only: politeness is out of scope, so this
entry is not in the shipped file):

```json
{
  "id": "ko-politeness-hasupnida",
  "lang": "ko",
  "distinction_class": "politeness",
  "match_type": "suffix",
  "surface_forms": ["습니다", "ㅂ니다"],
  "value": "formal",
  "notes": "Formal (hasipsio-che) verb ending allomorphs. value must be 'formal' or 'informal' -- see Closed-vocabulary classes."
}
```

## How to add entries

1. Append new objects to `entries`. No code change is needed unless you
   need a `match_type` this file doesn't already support (see "Match
   types").
2. Load and validate the file:

   ```python
   from dkmem.memory.lexicon import load_lexicon
   lex = load_lexicon("dkmem/memory/distinction_features.json")
   print(len(lex), "entries;", sorted(lex.distinction_classes))
   ```

   Any problem (bad JSON, a missing/unknown field, an invalid regex, a
   value outside a closed class's allowed set, a duplicate id, or a
   conflicting surface form) raises `LexiconError` naming the entry's index
   and, where available, its `id`.
3. Run `python -m unittest tests.test_lexicon -v` (or the full suite) —
   `test_lexicon.py` loads this exact file and checks it parses.

## Matching

Tokenizing an utterance and applying `exact_word`/`prefix`/`suffix`/`regex`
against it, and resolving overlapping matches with `priority`, is
implemented in `dkmem/memory/matcher.py` (`match_distinctions`), on top of
the `Lexicon` this file loads. See that module's docstring for the exact
tokenization and tie-break rules.

## What this format does *not* cover yet

- `name_variant` entries (see above) — in scope, not populated, and no
  matching approach is decided.
- Anything that needs sentence-level context to disambiguate, such as
  Hindi *kal*'s yesterday/tomorrow reading (which depends on verb tense, not
  just the word itself). That case is out of scope for the project; the
  matcher is deliberately context-free and cannot resolve such dependencies.
