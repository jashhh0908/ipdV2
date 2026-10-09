"""The host's LLM merge judge (Configs A, B, C of DKMEM_NEW_RESEARCH_IDEA.md Sec 6.2).

For one candidate pair of stored memory entries the judge answers "merge or
keep both?". It sees exactly the text each config stores (English gloss in A,
the extractor's facts in B, the verbatim utterance in C) and nothing else: no
distinction tags, no language label, and no hint about what kind of difference
matters. (A prompt that explains such differences to the judge is the separate
"prompt-informed judge" baseline, ``MERGE_JUDGE_INFORMED_V1``, further down: advice
the judge weighs, not a hard cannot-link rule.) Because the judge never sees the
DK-Mem metadata, its decision is the same whether the DK-Mem gate is on or off, and
the gate is applied afterwards (``dkmem.memory.gate.apply_gate``).

``MERGE_JUDGE_V1`` and ``MERGE_JUDGE_INFORMED_V1`` are frozen like the extraction
prompts; write a new id instead of editing either. The judge answers with exactly
``{"decision": "merge" | "keep_both", "reason": str}``. Anything else is a
``JudgeResult`` with ``error`` set and the raw output kept: a failed judgement
is never turned into a decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from dkmem.backends.llm import GenerationParams
from dkmem.memory.extract import TextGenerator
from dkmem.memory.prompts import PromptTemplate

__all__ = [
    "JUDGE_DECISIONS",
    "MERGE_JUDGE_V1",
    "MERGE_JUDGE_INFORMED_V1",
    "JUDGE_PROMPTS",
    "JudgeResult",
    "parse_judge_output",
    "render_judge_prompt",
    "judge_pairs",
]

JUDGE_DECISIONS = ("merge", "keep_both")

_MERGE_JUDGE_V1_SYSTEM = """\
You maintain the long-term memory of a personal assistant. Two memory entries were found to be similar. Decide whether they record the same thing, so that they should be merged into one entry, or different things, so that both should be kept.

Answer with exactly one JSON object with exactly these two keys and nothing else (no markdown, no code fences, no commentary):
{"decision": "merge" or "keep_both", "reason": string}

- "merge": both entries state the same fact about the same person, thing or event.
- "keep_both": the entries are about different people, things or events, or state different facts.
- "reason": one short sentence."""

# The prompt carries two plain-English examples as chat turns. Neither involves
# a kinship term, a register or a name, so they do not hint at the probe classes.
_JUDGE_EXAMPLES = (
    (
        "Memory A: User likes strong black coffee\nMemory B: User likes strong black coffee in the morning",
        {"decision": "merge", "reason": "Both state the same coffee preference."},
    ),
    (
        "Memory A: User has a meeting at 3pm on Monday\nMemory B: User is flying to Delhi on Friday",
        {"decision": "keep_both", "reason": "They are two unrelated events."},
    ),
)

MERGE_JUDGE_V1 = PromptTemplate(
    prompt_id="merge_judge_v1",
    system=_MERGE_JUDGE_V1_SYSTEM,
    user_template="{utterance}",  # the slot carries the two-entry block, see render_judge_prompt
    output_keys=("decision", "reason"),
    examples=tuple((q, json.dumps(a, ensure_ascii=False)) for q, a in _JUDGE_EXAMPLES),
)

# The prompt-informed judge baseline (DKMEM_NEW_RESEARCH_IDEA.md Sec 6.3): merge_judge_v1 plus ONE
# added paragraph that tells the judge cross-lingual distinctions exist. The paragraph explains and
# asks the judge to weigh the evidence; it is advice, not a rule -- the judge keeps the decision, so
# this is deliberately NOT a hard cannot-link constraint (that is what the DK-Mem gate adds). The
# system text before the paragraph, the two chat examples, the user template and the output format
# are byte-identical to merge_judge_v1, so the two prompts differ by this paragraph only. The
# illustrative terms (Hindi "devar" / "jeth") occur neither in the Tier 1 input nor in the lexicon.
_INFORMED_GUIDANCE = """\
Note on languages: entries may be written in, or translated from, languages other than English, including romanized Hindi (Hinglish). Other languages often distinguish things that English words do not, so two entries that look alike in English can concern different people or things. For example, Hindi "devar" (husband's younger brother) and "jeth" (husband's elder brother) are different relatives although English calls both "brother-in-law"; some languages mark formality or politeness in the pronoun or verb form; and similar-looking names can belong to different people. When two entries use different source-language terms of this kind, treat that as evidence that they may concern different people or things. Spelling or script variants of the same name or term are not such a difference. Weigh this evidence together with everything else in the entries."""

_MERGE_JUDGE_INFORMED_V1_SYSTEM = _MERGE_JUDGE_V1_SYSTEM + "\n\n" + _INFORMED_GUIDANCE

MERGE_JUDGE_INFORMED_V1 = PromptTemplate(
    prompt_id="merge_judge_informed_v1",
    system=_MERGE_JUDGE_INFORMED_V1_SYSTEM,
    user_template="{utterance}",
    output_keys=("decision", "reason"),
    examples=tuple((q, json.dumps(a, ensure_ascii=False)) for q, a in _JUDGE_EXAMPLES),
)

# Every judge prompt a run may use, by id (the CLIs take the id; run_config.json records id + sha256).
JUDGE_PROMPTS = {p.prompt_id: p for p in (MERGE_JUDGE_V1, MERGE_JUDGE_INFORMED_V1)}


@dataclass(frozen=True)
class JudgeResult:
    """One judgement. Exactly one of ``decision`` and ``error`` is set."""

    raw_output: str
    decision: str | None
    reason: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def render_judge_prompt(
    text_a: str, text_b: str, prompt: PromptTemplate = MERGE_JUDGE_V1
) -> list[dict[str, str]]:
    """Chat messages asking the judge about the stored texts of entries A and B."""
    return prompt.render(f"Memory A: {text_a}\nMemory B: {text_b}")


def parse_judge_output(raw_output: str) -> JudgeResult:
    """Strictly parse one judge response (no fence stripping, no repair)."""
    if not isinstance(raw_output, str):
        return JudgeResult(repr(raw_output), None, None, f"model output must be str, got {type(raw_output).__name__}")
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        return JudgeResult(raw_output, None, None, f"output is not a single JSON object ({e})")
    if not isinstance(data, dict) or set(data) != {"decision", "reason"}:
        return JudgeResult(raw_output, None, None, f"output must be an object with keys decision, reason; got {data!r}")
    decision, reason = data["decision"], data["reason"]
    if decision not in JUDGE_DECISIONS:
        return JudgeResult(raw_output, None, None, f"decision must be one of {JUDGE_DECISIONS}, got {decision!r}")
    if not isinstance(reason, str):
        return JudgeResult(raw_output, None, None, f"reason must be a string, got {type(reason).__name__}")
    return JudgeResult(raw_output, decision, reason)


def judge_pairs(
    pairs: Sequence[tuple[str, str]],
    generator: TextGenerator,
    params: GenerationParams | None = None,
    *,
    prompt: PromptTemplate = MERGE_JUDGE_V1,
) -> list[JudgeResult]:
    """Judge ``(text_a, text_b)`` pairs with one batched ``generate`` call.

    Results are in input order; a malformed response is a ``JudgeResult`` with
    ``error`` set rather than an exception.
    """
    params = params or GenerationParams()
    pairs = list(pairs)
    if not pairs:
        return []
    outputs = generator.generate([render_judge_prompt(a, b, prompt) for a, b in pairs], params)
    if not isinstance(outputs, list) or len(outputs) != len(pairs):
        got = len(outputs) if isinstance(outputs, list) else type(outputs).__name__
        raise ValueError(f"generator returned {got} outputs for {len(pairs)} prompts")
    return [parse_judge_output(raw) for raw in outputs]
