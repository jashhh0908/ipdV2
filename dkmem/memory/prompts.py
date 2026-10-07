"""Frozen extraction prompt definitions, keyed by ``prompt_id``.

A prompt's text must never change once experiments log its ``prompt_id``
(``Extraction.prompt_id`` / ``MergeEvent.prompt_id``); write a new id instead
(e.g. a native-language variant for the prompt-language ablation).
``PromptTemplate.sha256`` fingerprints the exact text for provenance.

No generation or parsing happens here; ``render`` returns chat messages that
``dkmem.backends.llm.HFGenerator.generate`` accepts as a prompt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "PromptTemplate",
    "MEM0_EXTRACTION_V1",
    "MEM0_EXTRACTION_V2",
    "MEM0_EXTRACTION_V3",
    "MEM0_EXTRACTION_V4",
    "DEFAULT_EXTRACTION_PROMPT",
    "PROMPTS",
    "get_prompt",
]


@dataclass(frozen=True)
class PromptTemplate:
    """An immutable system + user prompt pair with one ``{utterance}`` slot.

    ``output_keys`` are the exact top-level keys the model must return as a
    single JSON object. ``examples`` are optional ``(utterance, output)``
    few-shot pairs rendered as prior user/assistant chat turns.
    """

    prompt_id: str
    system: str
    user_template: str
    output_keys: tuple[str, ...]
    examples: tuple[tuple[str, str], ...] = ()

    def render(self, utterance: str) -> list[dict[str, str]]:
        """Chat messages for one utterance."""
        if not isinstance(utterance, str):
            raise TypeError("utterance must be str")
        messages = [{"role": "system", "content": self.system}]
        for example_utterance, example_output in self.examples:
            messages.append(
                {"role": "user", "content": self.user_template.format(utterance=example_utterance)}
            )
            messages.append({"role": "assistant", "content": example_output})
        messages.append({"role": "user", "content": self.user_template.format(utterance=utterance)})
        return messages

    @property
    def sha256(self) -> str:
        """Fingerprint of the exact prompt text (unchanged formula when no examples)."""
        parts = [self.prompt_id, self.system, self.user_template, *self.output_keys]
        for example_utterance, example_output in self.examples:
            parts += ["\x01example", example_utterance, example_output]
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


_MEM0_EXTRACTION_V1_SYSTEM = """\
You are the memory-extraction component of a personal assistant. You read one user utterance, which may be in any language or script, or code-mixed, and convert it into one memory entry.

Return exactly one JSON object with exactly these three keys and nothing else (no markdown, no code fences, no commentary):
{"gloss": string, "distinction": object, "surface": string}

"gloss":
- One short sentence in plain English stating the fact in the utterance.
- Write in the third person. Call the speaker "user" and the person being spoken to "addressee", both in lowercase. Call other people by name or by their relation to the user (e.g. "user's uncle").
- Use common, general English words only and no non-English words. Source-language detail that plain English does not express belongs in "distinction", not in the gloss.
- Resolve time words to their meaning in context (e.g. "yesterday", "tomorrow").
- Do not end with a period.

"distinction":
- An object recording source-language distinctions that the English gloss does not preserve. Use only the keys below, and include a key only when the utterance marks that feature. All values are strings. If nothing applies, use {}.
- "kinship": a kinship term used in a language other than English. Give the source-language term, romanized and lowercase, when English has no exact single-word equivalent (e.g. "mama", "bua"); otherwise give its English translation (e.g. "son"). An English kinship word used inside a non-English utterance (e.g. "aunty", "uncle") marks nothing: omit the key.
- "register": the pronoun or address form that marks formality, romanized and lowercase (e.g. "tu", "du", "sie").
- "politeness": the speech level encoded in the verb ending: "formal" or "informal".
- "evidentiality": how the speaker knows the fact: "direct/confirmed" or "reported/hearsay".
- "classifier": the category of the measure word or classifier, in English (e.g. "cup", "long-object", "flat-object").
- "temporal_deixis": for a time word that is ambiguous in the source language (e.g. Hindi "kal" means yesterday or tomorrow), the reading implied by context: "yesterday" or "tomorrow".
- "name_variant": for a person's name, "<Name>-<script>" with the name romanized and the script in lowercase (e.g. "Rahul-latin", "Rahul-devanagari").

"surface":
- The utterance copied exactly, character for character, in its original language and script.

Examples:

Utterance: Mera mama Delhi mein kaam karta hai.
{"gloss": "user's uncle works in Delhi", "distinction": {"kinship": "mama"}, "surface": "Mera mama Delhi mein kaam karta hai."}

Utterance: Main har subah gym jaata hoon.
{"gloss": "user goes to the gym every morning", "distinction": {}, "surface": "Main har subah gym jaata hoon."}"""

MEM0_EXTRACTION_V1 = PromptTemplate(
    prompt_id="mem0_extraction_v1",
    system=_MEM0_EXTRACTION_V1_SYSTEM,
    user_template="Utterance: {utterance}",
    output_keys=("gloss", "distinction", "surface"),
)

_MEM0_EXTRACTION_V2_SYSTEM = """\
You convert one user utterance into one memory entry for a personal assistant. The utterance may be in any language or script, or code-mixed.

Output exactly one JSON object with exactly these three keys and nothing else (no markdown, no code fences, no commentary):
{"gloss": string, "distinction": object, "surface": string}

## gloss
- One short, accurate English sentence stating the fact. Translate the meaning faithfully: do not add, drop, or reverse who did what to whom.
- Write in the third person. The speaker ("I", "me", "my" in any language) is "user"; the person spoken to ("you") is "addressee"; both in lowercase. Keep people's names.
- Use plain, common English words only. For a kinship term, use the general English word for that relative (e.g. "uncle", "aunt", "grandmother"). Detail that English does not express goes in "distinction", not in the gloss.
- Resolve time words to their meaning in context (e.g. "yesterday", "tomorrow", "last night").
- Do not end with a period.

## distinction
An object recording source-language features that the English gloss loses. Check every feature below and add its key only if the utterance really contains that feature. Several keys may apply. If none apply, use {}. All values are strings.

- "kinship": a word for a relative in a language other than English.
  Value: the source word, romanized and lowercase, when English has no exact single-word equivalent (e.g. Hindi "mama" = maternal uncle); otherwise its English translation (e.g. Spanish "hijo" -> "son").
  An English relative word used inside a non-English sentence (e.g. "uncle", "aunty") is not marked: omit the key.
- "register": the form of "you" chosen to address someone, showing familiarity or respect.
  Value: that pronoun, romanized and lowercase (e.g. Hindi "tu", German "du" or "sie", French "tu" or "vous").
  Only for a second-person pronoun that appears in the utterance. Never for verb endings (that is "politeness") and never for "I" or "my".
- "politeness": the speech level shown by the verb ending, as in Korean or Japanese.
  Value: "formal" (e.g. Korean -습니다/-ㅂ니다, Japanese -ます/-です) or "informal" (e.g. Korean plain -어/-아 endings, Japanese plain forms).
- "evidentiality": a grammatical marker of how the speaker knows the fact, as in Turkish.
  Value: "reported/hearsay" (e.g. Turkish -mış/-miş/-muş/-müş: heard from others or inferred) or "direct/confirmed" (e.g. Turkish -dı/-di/-du/-dü/-tı/-ti/-tu/-tü: witnessed).
- "classifier": a counter or measure word used with a number, as in Japanese, Chinese or Korean.
  Value: the counter's category, one of: "cup" (cups or glasses of drink), "long-object" (long, thin things), "flat-object" (thin, flat things), "small-object" (small or round things), "animal", "person", "machine", "volume" (books), "general" (general-purpose counter). Give the category, never the counted noun.
- "temporal_deixis": a time word that is ambiguous in the source language (e.g. Hindi/Urdu "kal" means yesterday or tomorrow).
  Value: the reading given by the verb tense: "yesterday" or "tomorrow". Not for unambiguous time words.
- "name_variant": a person's name.
  Value: "<Name>-<script>": the name, then the script it is written in, lowercase (e.g. "Rahul-latin", "Rahul-devanagari").

## surface
The utterance copied exactly, character for character, in its original language and script.

## Examples

Utterance: Mere mama Jaipur mein padhate hain.
{"gloss": "user's uncle teaches in Jaipur", "distinction": {"kinship": "mama"}, "surface": "Mere mama Jaipur mein padhate hain."}

Utterance: Mi hijo estudia en Madrid.
{"gloss": "user's son studies in Madrid", "distinction": {"kinship": "son"}, "surface": "Mi hijo estudia en Madrid."}

Utterance: Mere uncle Chennai mein rehte hain.
{"gloss": "user's uncle lives in Chennai", "distinction": {}, "surface": "Mere uncle Chennai mein rehte hain."}

Utterance: Tu bahut accha khana banata hai.
{"gloss": "addressee cooks very well", "distinction": {"register": "tu"}, "surface": "Tu bahut accha khana banata hai."}

Utterance: Du hast morgen frei.
{"gloss": "addressee has the day off tomorrow", "distinction": {"register": "du"}, "surface": "Du hast morgen frei."}

Utterance: 저는 내일 부산에 갑니다.
{"gloss": "user will go to Busan tomorrow", "distinction": {"politeness": "formal"}, "surface": "저는 내일 부산에 갑니다."}

Utterance: 나 어제 영화 봤어.
{"gloss": "user watched a movie yesterday", "distinction": {"politeness": "informal"}, "surface": "나 어제 영화 봤어."}

Utterance: Ayşe İzmir'e taşınmış.
{"gloss": "Ayşe moved to Izmir", "distinction": {"evidentiality": "reported/hearsay", "name_variant": "Ayşe-latin"}, "surface": "Ayşe İzmir'e taşınmış."}

Utterance: Mehmet dün eve geldi.
{"gloss": "Mehmet came home yesterday", "distinction": {"evidentiality": "direct/confirmed", "name_variant": "Mehmet-latin"}, "surface": "Mehmet dün eve geldi."}

Utterance: 切手を五枚買いました。
{"gloss": "user bought five stamps", "distinction": {"classifier": "flat-object", "politeness": "formal"}, "surface": "切手を五枚買いました。"}

Utterance: Maine kal pizza khaya.
{"gloss": "user ate pizza yesterday", "distinction": {"temporal_deixis": "yesterday"}, "surface": "Maine kal pizza khaya."}

Utterance: राहुल ने मुझे किताब दी।
{"gloss": "Rahul gave the user a book", "distinction": {"name_variant": "Rahul-devanagari"}, "surface": "राहुल ने मुझे किताब दी।"}

Utterance: Main har subah gym jaata hoon.
{"gloss": "user goes to the gym every morning", "distinction": {}, "surface": "Main har subah gym jaata hoon."}"""

MEM0_EXTRACTION_V2 = PromptTemplate(
    prompt_id="mem0_extraction_v2",
    system=_MEM0_EXTRACTION_V2_SYSTEM,
    user_template="Utterance: {utterance}",
    output_keys=("gloss", "distinction", "surface"),
)

_MEM0_EXTRACTION_V3_SYSTEM = """\
You convert one user utterance into one memory entry for a personal assistant. The utterance may be in any language or script, or code-mixed.

Output exactly one JSON object with exactly these three keys and nothing else (no markdown, no code fences, no commentary):
{"gloss": string, "distinction": object, "surface": string}

## gloss
- One short, accurate English sentence stating the fact. Translate the meaning faithfully: do not add, drop, or reverse who did what to whom.
- Write in the third person. The speaker ("I", "me", "my" in any language) is "user"; the person spoken to ("you") is "addressee"; both in lowercase. Never write "I", "me", "my" or "you" in the gloss. Keep people's names.
- Use plain, common English words only. For a kinship term, use the general English word for that relative (e.g. "uncle", "aunt", "grandmother"). Detail that English does not express goes in "distinction", not in the gloss.
- Resolve time words to their meaning in context (e.g. "yesterday", "tomorrow", "last night").
- Do not end with a period.

## distinction
An object recording source-language features that the English gloss loses. Most utterances mark zero or one feature. Add a key only when the utterance itself contains the marker described below; never carry a feature over from an earlier example. If none apply, use {}. All values are strings.

- "kinship": a word for a relative in a language other than English.
  Value: the source word, romanized and lowercase, when English has no exact single-word equivalent (e.g. Hindi "mama" = maternal uncle); otherwise its English translation (e.g. Spanish "hijo" -> "son").
  Omit it for an English relative word inside a non-English sentence (e.g. "uncle", "aunty"), and for titles, jobs or roles (e.g. teacher, doctor, boss).
- "register": the form of "you" chosen to address someone, showing familiarity or respect.
  Value: that pronoun, romanized and lowercase (e.g. Hindi "tu", German "du" or "sie", French "tu" or "vous").
  Only when a second-person pronoun is written in the utterance. Never for verb endings (that is "politeness") and never for "I" or "my".
- "politeness": the speech level shown by the verb ending, as in Korean or Japanese.
  Value: "formal" (e.g. Korean -습니다/-ㅂ니다, Japanese -ます/-です) or "informal" (e.g. Korean plain -어/-아 endings, Japanese plain forms).
- "evidentiality": a grammatical marker of how the speaker knows the fact, as in Turkish.
  Value: "reported/hearsay" (e.g. Turkish -mış/-miş/-muş/-müş: heard from others or inferred) or "direct/confirmed" (e.g. Turkish -dı/-di/-du/-dü/-tı/-ti/-tu/-tü: witnessed).
  Only for languages that mark this in grammar; never for Hindi, English or German.
- "classifier": a counter or measure word used with a number, as in Japanese, Chinese or Korean.
  Value: the counter's category, one of: "cup" (cups or glasses of drink), "long-object" (long, thin things), "flat-object" (thin, flat things), "small-object" (small or round things), "animal", "person", "machine", "volume" (books), "general" (general-purpose counter). Give the category, never the counted noun.
- "temporal_deixis": a time word that is ambiguous in the source language (e.g. Hindi/Urdu "kal" means yesterday or tomorrow, in any script).
  Value: the reading given by the verb tense: "yesterday" or "tomorrow". Not for unambiguous time words.
- "name_variant": the name of a specific person (never a pronoun such as "I" or "you").
  Value: "<Name>-<script>": the name, then the script it is actually written in, lowercase (e.g. "Rahul-latin" for Latin letters, "Rahul-devanagari" for Devanagari letters).

## surface
The utterance copied exactly, character for character, in its original language and script."""


def _example(utterance: str, gloss: str, distinction: dict[str, str]) -> tuple[str, str]:
    output = json.dumps(
        {"gloss": gloss, "distinction": distinction, "surface": utterance}, ensure_ascii=False
    )
    return (utterance, output)


MEM0_EXTRACTION_V3 = PromptTemplate(
    prompt_id="mem0_extraction_v3",
    system=_MEM0_EXTRACTION_V3_SYSTEM,
    user_template="Utterance: {utterance}",
    output_keys=("gloss", "distinction", "surface"),
    examples=(
        _example("Mere mama Jaipur mein padhate hain.", "user's uncle teaches in Jaipur", {"kinship": "mama"}),
        _example("Mi hijo estudia en Madrid.", "user's son studies in Madrid", {"kinship": "son"}),
        _example("Mere uncle Chennai mein rehte hain.", "user's uncle lives in Chennai", {}),
        _example("Tu bahut accha khana banata hai.", "addressee cooks very well", {"register": "tu"}),
        _example("Du hast morgen frei.", "addressee has the day off tomorrow", {"register": "du"}),
        _example("저는 내일 부산에 갑니다.", "user will go to Busan tomorrow", {"politeness": "formal"}),
        _example("나 어제 영화 봤어.", "user watched a movie yesterday", {"politeness": "informal"}),
        _example("Ayşe İzmir'e taşınmış.", "Ayşe moved to Izmir",
                 {"evidentiality": "reported/hearsay", "name_variant": "Ayşe-latin"}),
        _example("Mehmet dün eve geldi.", "Mehmet came home yesterday",
                 {"evidentiality": "direct/confirmed", "name_variant": "Mehmet-latin"}),
        _example("切手を五枚買いました。", "user bought five stamps",
                 {"classifier": "flat-object", "politeness": "formal"}),
        _example("Maine kal pizza khaya.", "user ate pizza yesterday", {"temporal_deixis": "yesterday"}),
        _example("राहुल ने मुझे किताब दी।", "Rahul gave the user a book", {"name_variant": "Rahul-devanagari"}),
        _example("Main har subah gym jaata hoon.", "user goes to the gym every morning", {}),
    ),
)

# v4: the final prompt, re-scoped to the current research plan. It is v3 (the
# best-performing prompt: same 10/20 target-feature accuracy as v2 but no
# "you"/"My ..." gloss violations) restricted to the in-scope distinction
# classes (kinship, register, name_variant; see dkmem.memory.scope). Removed
# from v3: the politeness, evidentiality, classifier and temporal_deixis
# definitions and the examples that only demonstrated them (the Turkish
# examples stay, tagging only the name). One addition: "Use only the keys
# below" (as in v1), because the parser rejects any other key outright. v1-v3
# are kept unchanged for comparison and still describe the old 7-class scope.
# Like v3, the examples are passed as chat turns. v4 has not been run on a
# model yet.
_MEM0_EXTRACTION_V4_SYSTEM = """\
You convert one user utterance into one memory entry for a personal assistant. The utterance may be in any language or script, or code-mixed.

Output exactly one JSON object with exactly these three keys and nothing else (no markdown, no code fences, no commentary):
{"gloss": string, "distinction": object, "surface": string}

## gloss
- One short, accurate English sentence stating the fact. Translate the meaning faithfully: do not add, drop, or reverse who did what to whom.
- Write in the third person. The speaker ("I", "me", "my" in any language) is "user"; the person spoken to ("you") is "addressee"; both in lowercase. Never write "I", "me", "my" or "you" in the gloss. Keep people's names.
- Use plain, common English words only. For a kinship term, use the general English word for that relative (e.g. "uncle", "aunt", "grandmother"). Detail that English does not express goes in "distinction", not in the gloss.
- Resolve time words to their meaning in context (e.g. "yesterday", "tomorrow", "last night").
- Do not end with a period.

## distinction
An object recording source-language features that the English gloss loses. Use only the keys below. Most utterances mark zero or one feature. Add a key only when the utterance itself contains the marker described below; never carry a feature over from an earlier example. If none apply, use {}. All values are strings.

- "kinship": a word for a relative in a language other than English.
  Value: the source word, romanized and lowercase, when English has no exact single-word equivalent (e.g. Hindi "mama" = maternal uncle); otherwise its English translation (e.g. Spanish "hijo" -> "son").
  Omit it for an English relative word inside a non-English sentence (e.g. "uncle", "aunty"), and for titles, jobs or roles (e.g. teacher, doctor, boss).
- "register": the form of "you" chosen to address someone, showing familiarity or respect.
  Value: that pronoun, romanized and lowercase (e.g. Hindi "tu", German "du" or "sie", French "tu" or "vous").
  Only when a second-person pronoun is written in the utterance. Never for verb endings and never for "I" or "my".
- "name_variant": the name of a specific person (never a pronoun such as "I" or "you").
  Value: "<Name>-<script>": the name, then the script it is actually written in, lowercase (e.g. "Rahul-latin" for Latin letters, "Rahul-devanagari" for Devanagari letters).

## surface
The utterance copied exactly, character for character, in its original language and script."""

MEM0_EXTRACTION_V4 = PromptTemplate(
    prompt_id="mem0_extraction_v4",
    system=_MEM0_EXTRACTION_V4_SYSTEM,
    user_template="Utterance: {utterance}",
    output_keys=("gloss", "distinction", "surface"),
    examples=(
        _example("Mere mama Jaipur mein padhate hain.", "user's uncle teaches in Jaipur", {"kinship": "mama"}),
        _example("Mi hijo estudia en Madrid.", "user's son studies in Madrid", {"kinship": "son"}),
        _example("Mere uncle Chennai mein rehte hain.", "user's uncle lives in Chennai", {}),
        _example("Tu bahut accha khana banata hai.", "addressee cooks very well", {"register": "tu"}),
        _example("Du hast morgen frei.", "addressee has the day off tomorrow", {"register": "du"}),
        _example("Ayşe İzmir'e taşınmış.", "Ayşe moved to Izmir", {"name_variant": "Ayşe-latin"}),
        _example("Mehmet dün eve geldi.", "Mehmet came home yesterday", {"name_variant": "Mehmet-latin"}),
        _example("राहुल ने मुझे किताब दी।", "Rahul gave the user a book", {"name_variant": "Rahul-devanagari"}),
        _example("Main har subah gym jaata hoon.", "user goes to the gym every morning", {}),
    ),
)

PROMPTS = MappingProxyType(
    {
        p.prompt_id: p
        for p in (MEM0_EXTRACTION_V1, MEM0_EXTRACTION_V2, MEM0_EXTRACTION_V3, MEM0_EXTRACTION_V4)
    }
)

# The prompt used when a caller does not pass one. Results record their own
# prompt_id, so runs made with v1 stay interpretable.
DEFAULT_EXTRACTION_PROMPT = MEM0_EXTRACTION_V4


def get_prompt(prompt_id: str) -> PromptTemplate:
    """Look up a frozen prompt by id."""
    try:
        return PROMPTS[prompt_id]
    except KeyError:
        raise KeyError(f"unknown prompt_id {prompt_id!r}; known: {sorted(PROMPTS)}") from None
