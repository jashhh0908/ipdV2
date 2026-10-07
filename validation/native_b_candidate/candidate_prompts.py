"""Candidate Config B prompt (native-language extraction). NOT part of dkmem/ and not frozen.

Config B must isolate L2 (language drift): the extractor is *told* to keep the
user's language and script, and the experiment measures whether it does and
whether the distinction survives. It must not be the Mem0 baseline, so it is our
own prompt with one fact per utterance and no fact-selection filter.

The language sentence is Mem0's, verbatim (singular "fact" instead of "facts");
the rest makes the instruction explicit about not translating and about keeping
names and relationship terms. That is a stronger instruction than Mem0's alone,
so any drift that remains is drift despite a clear instruction.
"""

from dkmem.memory.prompts import PromptTemplate

SYSTEM = """\
You convert one user utterance into one memory entry for a personal assistant. The utterance may be in any language or script, or code-mixed.

Output exactly one JSON object with exactly one key and nothing else (no markdown, no code fences, no commentary):
{"fact": string}

"fact":
- Exactly one short sentence stating the fact in the utterance. Always return one fact; never return an empty string.
- You should detect the language of the user input and record the fact in the same language.
- Keep the language and the script of the utterance. Do not translate it into English or into any other language. If the utterance is code-mixed, keep the mix.
- Keep names and relationship terms exactly as written in the utterance. Do not replace a relationship term with a more general word.
- Do not add information that is not in the utterance. Do not explain."""

NATIVE_B_CANDIDATE = PromptTemplate(
    prompt_id="native_b_candidate_v1",
    system=SYSTEM,
    user_template="Utterance: {utterance}",
    output_keys=("fact",),
)
