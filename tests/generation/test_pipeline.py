from types import SimpleNamespace

from libs.core.config import groundedness_settings
from libs.core.models import ChunkType
from libs.generation import pipeline as pipeline_module
from libs.generation.generator import Claim, GeneratedAnswer
from libs.generation.pipeline import answer
from libs.retrieval.pipeline import RetrievedResult


def _result(chunk_id: str, text: str, section_item_code: str = "7") -> RetrievedResult:
    return RetrievedResult(
        score=0.9,
        payload={
            "chunk_id": chunk_id,
            "filing_id": "acme-10k",
            "company": "Acme Robotics, Inc.",
            "form_type": "10-K",
            "filing_date": None,
            "section_item_code": section_item_code,
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
    """Returns one queued response per `.parse()` call: first the
    generator's `GeneratedAnswer`, then one groundedness verdict per claim,
    in the order `answer()` actually calls them."""

    def __init__(self, responses: list):
        self._responses = list(responses)

    def parse(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self._responses.pop(0)))]
        )


class _FakeOpenAIClient:
    def __init__(self, responses: list):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(responses))


class _Verdict:
    def __init__(self, supported: bool, confidence: float):
        self.supported = supported
        self.confidence = confidence


def test_answer_refuses_immediately_when_retrieval_finds_nothing(monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: [])

    result = answer("irrelevant question", None, None, None, None)

    assert result.answered is False
    assert result.groundedness is None
    assert "no relevant sources" in result.refusal_reason.lower()


def test_answer_returns_grounded_answer_with_deduped_citations(monkeypatch):
    results = [
        _result("acme-10k:7:0", "Revenue was $1B in 2025."),
        _result("acme-10k:7:1", "Headcount grew 5%."),
    ]
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: results)

    generated = GeneratedAnswer(
        answer="Revenue was $1B and headcount grew 5%.",
        claims=[
            Claim(text="Revenue was $1B.", chunk_id="acme-10k:7:0"),
            Claim(text="Headcount grew 5%.", chunk_id="acme-10k:7:0"),  # repeat chunk_id on purpose
        ],
    )
    openai_client = _FakeOpenAIClient(
        [generated, _Verdict(True, 0.9), _Verdict(True, 0.85)]
    )

    result = answer("How did the company perform?", None, openai_client, None, None)

    assert result.answered is True
    assert result.answer == "Revenue was $1B and headcount grew 5%."
    # Two claims cited the same chunk_id -> one citation, not two.
    assert len(result.citations) == 1
    assert result.citations[0]["chunk_id"] == "acme-10k:7:0"
    assert result.groundedness.overall_confidence == 0.85


def test_answer_refuses_when_groundedness_check_fails(monkeypatch):
    results = [_result("acme-10k:7:0", "Revenue was $1B in 2025.")]
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: results)

    generated = GeneratedAnswer(
        answer="Revenue was $2B.",
        claims=[Claim(text="Revenue was $2B.", chunk_id="acme-10k:7:0")],
    )
    openai_client = _FakeOpenAIClient([generated, _Verdict(False, 0.9)])

    result = answer("What was revenue?", None, openai_client, None, None)

    assert result.answered is False
    assert result.answer is None
    assert result.citations is None
    assert result.refusal_reason == groundedness_settings.refusal_message
    assert result.groundedness.is_grounded is False


def test_answer_refuses_when_confidence_is_below_threshold_even_if_supported(monkeypatch):
    """A claim can be marked `supported=True` by the judge but with low
    confidence -- the threshold check is a separate, additional guard on
    top of the binary supported/unsupported verdict, not a redundant one.
    """
    results = [_result("acme-10k:7:0", "Revenue was roughly $1B.")]
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: results)

    generated = GeneratedAnswer(
        answer="Revenue was $1B.",
        claims=[Claim(text="Revenue was $1B.", chunk_id="acme-10k:7:0")],
    )
    low_confidence = groundedness_settings.confidence_threshold - 0.1
    openai_client = _FakeOpenAIClient([generated, _Verdict(True, low_confidence)])

    result = answer("What was revenue?", None, openai_client, None, None)

    assert result.answered is False
