"""The groundedness gate: verifies every claim in a generated answer is
actually supported by its cited source before the answer is ever returned
to a user. This is the literal "cite or refuse" of the project.

Citation-rebind (added after a live investigation, 2026-07-25): generation
occasionally mis-binds a factually-correct, well-grounded claim to the
wrong chunk_id when multiple retrieved chunks discuss the same topic at
different levels of detail -- confirmed via 20 repeated runs of an
identical query against an identical index (retrieval and claim *content*
were 100% stable across runs; only the chunk_id a claim got attached to
occasionally varied). When a claim fails against its cited chunk, this
module now checks whether some *other already-retrieved* chunk supports
it before refusing, and re-attributes the claim to that chunk if so.

Known residual risk, not solved by this: rebind only fires when the
originally-cited chunk FAILS its groundedness check. A mis-bind to a
chunk that only loosely/imprecisely supports the same claim -- close
enough that the judge marks it `supported=True` -- never reaches the
rebind path at all, since nothing failed. That kind of imprecise-but-not-
wrong binding is invisible to this fix and to the eval harness's current
metrics; flagged here and in `data/eval/sec_filings_eval_set.json`, not
addressed.
"""
from openai import OpenAI
from pydantic import BaseModel, Field

from libs.core.config import groundedness_settings
from libs.core.logging import get_logger
from libs.generation.generator import Claim, GeneratedAnswer
from libs.retrieval.pipeline import RetrievedResult

logger = get_logger(__name__)

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
    citation check or the LLM-judge semantic check.

    `rebound_from_chunk_id` is set only when this verdict is the result of
    a successful citation-rebind: the claim's original `chunk_id` (here)
    failed its groundedness check, and `claim.chunk_id` was corrected to a
    different retrieved chunk that does support it.
    """

    claim: Claim
    supported: bool
    confidence: float
    reason: str | None = None
    rebound_from_chunk_id: str | None = None


class GroundednessResult(BaseModel):
    is_grounded: bool
    overall_confidence: float
    claim_verdicts: list[ClaimVerdict]
    rebind_count: int = 0
    """How many claims needed citation-rebind to pass. Telemetry, not just
    a debugging detail -- track this over time as an early-warning signal
    if citation-binding quality degrades (e.g. after a prompt or model
    change), even while `is_grounded` stays healthy because rebind quietly
    compensates for it."""


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


def _find_rebind(
    claim: Claim, results: list[RetrievedResult], exclude_chunk_id: str, client: OpenAI
) -> ClaimVerdict | None:
    """Searches already-retrieved chunks (skipping the one that just
    failed) for one that actually supports `claim`, trying `results` in
    the descending-relevance order they're already in and returning the
    first supporting verdict found.

    Never checks, cites, or invents anything outside `results` -- only
    re-attributes a claim among sources that were genuinely retrieved for
    this query.
    """
    for result in results:
        candidate_chunk_id = result.payload["chunk_id"]
        if candidate_chunk_id == exclude_chunk_id:
            continue
        candidate_claim = claim.model_copy(update={"chunk_id": candidate_chunk_id})
        verdict = _judge_claim(candidate_claim, result.payload["text"], client)
        if verdict.supported:
            return verdict
    return None


def check_groundedness(
    generated: GeneratedAnswer, results: list[RetrievedResult], client: OpenAI
) -> GroundednessResult:
    """Verifies every claim in `generated.claims`:

    1. Mechanical, no LLM call: does `claim.chunk_id` actually appear among
       the retrieved `results`? A citation to a chunk that was never
       retrieved is an automatic failure, NOT eligible for rebind (see
       module docstring) -- there's no need to ask an LLM whether a source
       supports a claim when that "source" was invented (a hallucinated
       citation, the failure mode a citation-format requirement alone
       can't catch, since the format can be followed syntactically while
       pointing at nothing real), and rebind exists to correct
       mis-attribution among genuinely retrieved chunks, not to rescue a
       fabricated reference by finding *some* real chunk that happens to
       also support the claim.
    2. Semantic, LLM-as-judge: for claims that pass (1), does the cited
       chunk's actual text support the claim, or does the claim overreach
       it? If not, before failing the claim outright, `_find_rebind` checks
       whether a *different* retrieved chunk supports it and re-attributes
       the claim there if so.

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

        verdict = _judge_claim(claim, source_payload["text"], client)
        if not verdict.supported:
            rebind = _find_rebind(claim, results, exclude_chunk_id=claim.chunk_id, client=client)
            if rebind is not None:
                logger.info(
                    "Rebound claim %r: chunk_id=%s did not support it, chunk_id=%s does",
                    claim.text,
                    claim.chunk_id,
                    rebind.claim.chunk_id,
                )
                verdict = rebind.model_copy(update={"rebound_from_chunk_id": claim.chunk_id})
        verdicts.append(verdict)

    is_grounded = all(v.supported for v in verdicts)
    overall_confidence = min(v.confidence for v in verdicts)
    rebind_count = sum(1 for v in verdicts if v.rebound_from_chunk_id is not None)

    return GroundednessResult(
        is_grounded=is_grounded,
        overall_confidence=overall_confidence,
        claim_verdicts=verdicts,
        rebind_count=rebind_count,
    )
