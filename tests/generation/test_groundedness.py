from types import SimpleNamespace

from libs.core.models import ChunkType
from libs.generation.generator import Claim, GeneratedAnswer
from libs.generation.groundedness import check_groundedness
from libs.retrieval.pipeline import RetrievedResult


def _result(chunk_id: str, text: str) -> RetrievedResult:
    return RetrievedResult(
        score=0.9,
        payload={
            "chunk_id": chunk_id,
            "filing_id": "acme-10k",
            "company": "Acme Robotics, Inc.",
            "form_type": "10-K",
            "filing_date": None,
            "section_item_code": "7",
            "section_title": "MD&A",
            "part": "II",
            "chunk_type": ChunkType.TEXT.value,
            "order_index": 0,
            "char_start": 0,
            "char_end": len(text),
            "token_count": len(text.split()),
            "text": text,
        },
    )


class _FakeChatCompletions:
    """Returns one queued verdict per `.parse()` call, in order."""

    def __init__(self, verdicts: list):
        self._verdicts = list(verdicts)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self._verdicts.pop(0)))]
        )


class _FakeOpenAIClient:
    def __init__(self, verdicts: list):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(verdicts))


class _Verdict:
    def __init__(self, supported: bool, confidence: float):
        self.supported = supported
        self.confidence = confidence


def test_claim_citing_an_unretrieved_chunk_fails_without_calling_the_llm():
    """A hallucinated citation -- a chunk_id the model made up, that was
    never actually retrieved -- must be caught mechanically. No LLM call
    should even be needed to know a source that doesn't exist can't
    support anything.
    """
    generated = GeneratedAnswer(
        answer="Revenue was $1B.",
        claims=[Claim(text="Revenue was $1B.", chunk_id="acme-10k:7:999")],
    )
    results = [_result("acme-10k:7:0", "Revenue was $1B in 2025.")]
    client = _FakeOpenAIClient(verdicts=[])  # would raise IndexError if called

    result = check_groundedness(generated, results, client)

    assert result.is_grounded is False
    assert result.claim_verdicts[0].supported is False
    assert "not among the retrieved sources" in result.claim_verdicts[0].reason
    assert client.chat.completions.calls == []


def test_claim_supported_by_its_source_is_grounded():
    generated = GeneratedAnswer(
        answer="Revenue was $1B.",
        claims=[Claim(text="Revenue was $1B.", chunk_id="acme-10k:7:0")],
    )
    results = [_result("acme-10k:7:0", "Revenue was $1B in fiscal 2025.")]
    client = _FakeOpenAIClient(verdicts=[_Verdict(supported=True, confidence=0.92)])

    result = check_groundedness(generated, results, client)

    assert result.is_grounded is True
    assert result.overall_confidence == 0.92


def test_claim_not_supported_by_its_source_is_not_grounded():
    generated = GeneratedAnswer(
        answer="Revenue was $2B.",
        claims=[Claim(text="Revenue was $2B.", chunk_id="acme-10k:7:0")],
    )
    results = [_result("acme-10k:7:0", "Revenue was $1B in fiscal 2025.")]
    client = _FakeOpenAIClient(verdicts=[_Verdict(supported=False, confidence=0.85)])

    result = check_groundedness(generated, results, client)

    assert result.is_grounded is False


def test_overall_confidence_is_the_minimum_across_claims_not_the_average():
    generated = GeneratedAnswer(
        answer="Revenue was $1B and headcount grew.",
        claims=[
            Claim(text="Revenue was $1B.", chunk_id="acme-10k:7:0"),
            Claim(text="Headcount grew.", chunk_id="acme-10k:7:1"),
        ],
    )
    results = [
        _result("acme-10k:7:0", "Revenue was $1B."),
        _result("acme-10k:7:1", "Headcount increased this year."),
    ]
    client = _FakeOpenAIClient(
        verdicts=[
            _Verdict(supported=True, confidence=0.95),
            _Verdict(supported=True, confidence=0.6),
        ]
    )

    result = check_groundedness(generated, results, client)

    assert result.overall_confidence == 0.6  # min, not (0.95+0.6)/2


def test_a_single_unsupported_claim_sinks_an_otherwise_grounded_answer():
    generated = GeneratedAnswer(
        answer="Revenue was $1B and profit doubled.",
        claims=[
            Claim(text="Revenue was $1B.", chunk_id="acme-10k:7:0"),
            Claim(text="Profit doubled.", chunk_id="acme-10k:7:1"),
        ],
    )
    results = [
        _result("acme-10k:7:0", "Revenue was $1B."),
        _result("acme-10k:7:1", "Profit was roughly flat year over year."),
    ]
    client = _FakeOpenAIClient(
        verdicts=[
            _Verdict(supported=True, confidence=0.95),
            _Verdict(supported=False, confidence=0.9),  # claim 2 vs its own chunk
            _Verdict(supported=False, confidence=0.3),  # rebind attempt vs the other chunk -- also fails
        ]
    )

    result = check_groundedness(generated, results, client)

    assert result.is_grounded is False
    assert result.rebind_count == 0  # rebind was attempted but found nothing to rebind to


def test_rebind_corrects_a_claim_misbound_to_the_wrong_retrieved_chunk():
    """Deterministic reproduction of the real failure found in Phase 5's
    Tesla flakiness investigation (run 6 of 20): generation produced a
    factually-correct claim, word for word, but attached it to
    tsla-2025-10k:1:0 (Item 1 Business overview -- doesn't contain the
    detail) instead of tsla-2025-10k:8:253 (Note 16, Segment Reporting --
    does, verbatim). Deliberately does not rely on reproducing the ~5%
    flake live; this pins the exact scenario down with fakes.
    """
    claim = Claim(
        text=(
            "The energy generation and storage segment includes the design, "
            "manufacture, installation, sales and leasing of energy generation "
            "and storage products and related services."
        ),
        chunk_id="tsla-2025-10k:1:0",  # wrong -- this is the bug being fixed
    )
    generated = GeneratedAnswer(answer="...", claims=[claim])
    results = [
        _result("tsla-2025-10k:1:0", "ITEM 1. BUSINESS\n\nOverview\n\nWe design, develop, manufacture..."),
        _result(
            "tsla-2025-10k:8:253",
            "Note 16 - Segment Reporting...energy generation and storage segment includes the design, "
            "manufacture, installation, sales and leasing of energy generation and storage products...",
        ),
    ]
    client = _FakeOpenAIClient(
        verdicts=[
            _Verdict(supported=False, confidence=0.9),  # vs the wrong chunk (1:0) -- fails, as it should
            _Verdict(supported=True, confidence=1.0),  # vs the right chunk (8:253) -- rebind succeeds
        ]
    )

    result = check_groundedness(generated, results, client)

    assert result.is_grounded is True
    assert result.rebind_count == 1
    verdict = result.claim_verdicts[0]
    assert verdict.supported is True
    assert verdict.claim.chunk_id == "tsla-2025-10k:8:253"  # corrected
    assert verdict.rebound_from_chunk_id == "tsla-2025-10k:1:0"  # original, wrong


def test_rebind_never_applies_to_a_hallucinated_citation():
    """A chunk_id that was never retrieved at all is a stronger, different
    failure signal than mis-binding among real candidates -- rebind must
    not "rescue" it by finding some other real chunk that happens to
    support the same claim. Same zero-LLM-call guarantee as before.
    """
    generated = GeneratedAnswer(
        answer="Revenue was $1B.",
        claims=[Claim(text="Revenue was $1B.", chunk_id="acme-10k:7:999")],
    )
    results = [
        _result("acme-10k:7:0", "Revenue was $1B in 2025."),
        _result("acme-10k:7:1", "Revenue was $1B in 2025, per the summary table."),
    ]
    client = _FakeOpenAIClient(verdicts=[])  # would raise IndexError if called

    result = check_groundedness(generated, results, client)

    assert result.is_grounded is False
    assert result.rebind_count == 0
    assert client.chat.completions.calls == []


def test_rebind_tries_candidates_in_descending_relevance_order():
    """`results` is already rank-ordered; rebind should try candidates in
    that order and stop at the first that supports the claim, not
    exhaustively check every retrieved chunk.
    """
    claim = Claim(text="Segments are automotive and energy.", chunk_id="tsla:1:0")
    generated = GeneratedAnswer(answer="...", claims=[claim])
    results = [
        _result("tsla:1:0", "Overview text, no segment detail."),
        _result("tsla:8:253", "Segments are automotive and energy, in detail."),
        _result("tsla:7:60", "Also mentions automotive and energy segments."),
    ]
    client = _FakeOpenAIClient(
        verdicts=[
            _Verdict(supported=False, confidence=0.9),  # original chunk fails
            _Verdict(supported=True, confidence=0.99),  # first candidate (tsla:8:253) succeeds
            # a third verdict for tsla:7:60 is deliberately NOT queued --
            # if the search doesn't stop early, this test fails with IndexError
        ]
    )

    result = check_groundedness(generated, results, client)

    assert result.claim_verdicts[0].claim.chunk_id == "tsla:8:253"
    assert len(client.chat.completions.calls) == 2


def test_an_answer_with_no_claims_is_never_grounded():
    """An "answer" that cites nothing has nothing for this gate to verify
    -- that's a failure mode (the generator should have refused itself),
    not a trivially-passing edge case.
    """
    generated = GeneratedAnswer(answer="I don't have enough information.", claims=[])
    client = _FakeOpenAIClient(verdicts=[])

    result = check_groundedness(generated, [], client)

    assert result.is_grounded is False
    assert result.overall_confidence == 0.0
    assert client.chat.completions.calls == []
