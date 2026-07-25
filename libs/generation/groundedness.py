"""The groundedness gate: verifies every claim in a generated answer is
actually supported by its cited source before the answer is ever returned
to a user. This is the literal "cite or refuse" of the project.
"""
from openai import OpenAI
from pydantic import BaseModel, Field

from libs.core.config import groundedness_settings
from libs.generation.generator import Claim, GeneratedAnswer
from libs.retrieval.pipeline import RetrievedResult

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checker. Given a SOURCE passage and a CLAIM, judge "
    "whether the SOURCE actually supports the CLAIM -- not whether the CLAIM "
    "sounds plausible in general, and not whether it merely doesn't contradict "
    "the SOURCE. The CLAIM must not go beyond what the SOURCE literally says. "
    "Respond with `supported` (true only if the source directly supports the "
    "claim) and `confidence` (0.0-1.0, your confidence in that verdict)."
)


class _JudgeVerdict(BaseModel):
    supported: bool
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimVerdict(BaseModel):
    """One claim's groundedness verdict, either from the cheap mechanical
    citation check or the LLM-judge semantic check."""

    claim: Claim
    supported: bool
    confidence: float
    reason: str | None = None


class GroundednessResult(BaseModel):
    is_grounded: bool
    overall_confidence: float
    claim_verdicts: list[ClaimVerdict]


def _judge_claim(claim: Claim, source_text: str, client: OpenAI) -> ClaimVerdict:
    completion = client.chat.completions.parse(
        model=groundedness_settings.judge_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCE:\n{source_text}\n\nCLAIM:\n{claim.text}"},
        ],
        response_format=_JudgeVerdict,
    )
    verdict = completion.choices[0].message.parsed
    return ClaimVerdict(claim=claim, supported=verdict.supported, confidence=verdict.confidence)


def check_groundedness(
    generated: GeneratedAnswer, results: list[RetrievedResult], client: OpenAI
) -> GroundednessResult:
    """Verifies every claim in `generated.claims` in two stages:

    1. Mechanical, no LLM call: does `claim.chunk_id` actually appear among
       the retrieved `results`? A citation to a chunk that was never
       retrieved is an automatic failure -- there's no need to ask an LLM
       whether a source supports a claim when that "source" was invented
       (a hallucinated citation, the failure mode a citation-format
       requirement alone can't catch, since the format can be followed
       syntactically while pointing at nothing real).
    2. Semantic, LLM-as-judge: for claims that pass (1), does the cited
       chunk's actual text support the claim, or does the claim overreach it?

    An answer with zero claims is never grounded -- an "answer" that cites
    nothing has nothing for this gate to verify, which is a failure mode
    (the generator should have refused itself, via an empty `claims` list
    with an explanatory `answer`), not a trivially-passing edge case.

    `overall_confidence` is the *minimum* across claims, not an average --
    conservative on purpose, since one unsupported or low-confidence claim
    should be able to sink the whole answer's trustworthiness rather than
    being averaged away by several confident ones.
    """
    payload_by_chunk_id = {r.payload["chunk_id"]: r.payload for r in results}

    if not generated.claims:
        return GroundednessResult(is_grounded=False, overall_confidence=0.0, claim_verdicts=[])

    verdicts: list[ClaimVerdict] = []
    for claim in generated.claims:
        source_payload = payload_by_chunk_id.get(claim.chunk_id)
        if source_payload is None:
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    supported=False,
                    confidence=1.0,
                    reason="cited chunk_id was not among the retrieved sources",
                )
            )
            continue
        verdicts.append(_judge_claim(claim, source_payload["text"], client))

    is_grounded = all(v.supported for v in verdicts)
    overall_confidence = min(v.confidence for v in verdicts)

    return GroundednessResult(
        is_grounded=is_grounded,
        overall_confidence=overall_confidence,
        claim_verdicts=verdicts,
    )
