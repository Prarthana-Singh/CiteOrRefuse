from types import SimpleNamespace

from libs.core.config import generation_settings
from libs.generation.generator import Claim, GeneratedAnswer, generate_answer


class _FakeChatCompletions:
    def __init__(self, parsed_response: GeneratedAnswer):
        self._parsed_response = parsed_response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self._parsed_response))]
        )


class _FakeOpenAIClient:
    def __init__(self, parsed_response: GeneratedAnswer):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(parsed_response))


def test_generate_answer_returns_the_parsed_structured_response():
    expected = GeneratedAnswer(
        answer="Revenue grew 12% in 2025.",
        claims=[Claim(text="Revenue grew 12% in 2025.", chunk_id="acme-10k:7:0")],
    )
    client = _FakeOpenAIClient(expected)

    result = generate_answer("How did revenue grow?", "[chunk_id: acme-10k:7:0] ...", client)

    assert result == expected


def test_generate_answer_passes_context_and_query_to_the_llm():
    client = _FakeOpenAIClient(GeneratedAnswer(answer="x", claims=[]))
    context = "[chunk_id: acme-10k:7:0] Revenue grew 12%."

    generate_answer("How did revenue grow?", context, client)

    call = client.chat.completions.calls[0]
    assert call["model"] == generation_settings.model
    assert call["response_format"] is GeneratedAnswer
    user_message = call["messages"][-1]["content"]
    assert context in user_message
    assert "How did revenue grow?" in user_message


def test_generate_answer_system_prompt_mandates_citation_and_forbids_outside_knowledge():
    client = _FakeOpenAIClient(GeneratedAnswer(answer="x", claims=[]))

    generate_answer("q", "context", client)

    system_message = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "chunk_id" in system_message
    assert "outside knowledge" in system_message.lower() or "do not guess" in system_message.lower()
