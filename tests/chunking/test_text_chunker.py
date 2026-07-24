from libs.core.config import settings
from libs.chunking.strategies.text_chunker import chunk_text_blocks

FILING_ID = "acme-2023-10k"


def test_short_paragraphs_pack_into_a_single_chunk():
    paragraphs = [
        "Acme Robotics designs autonomous warehouse robots.",
        "We sell hardware, subscriptions, and maintenance contracts.",
    ]
    chunks = chunk_text_blocks(paragraphs, FILING_ID, "1", "Business", 0)

    assert len(chunks) == 1
    assembled = "\n\n".join(paragraphs)
    assert chunks[0].text == assembled
    assert assembled[chunks[0].char_start : chunks[0].char_end] == chunks[0].text


def test_many_paragraphs_split_into_overlapping_chunks():
    sentence = (
        "The Sentinel robot platform continues to expand across distribution centers "
        "operated by our largest logistics and e-commerce customers worldwide."
    )
    paragraphs = [f"{sentence} Paragraph number {i}." for i in range(20)]

    chunks = chunk_text_blocks(paragraphs, FILING_ID, "7", "MD&A", 0)

    assert len(chunks) > 1
    assert all(c.token_count <= settings.max_tokens for c in chunks)

    assembled = "\n\n".join(paragraphs)
    for chunk in chunks:
        assert assembled[chunk.char_start : chunk.char_end] == chunk.text

    # Consecutive chunks should overlap: the next chunk's start text should
    # already appear near the end of the previous chunk.
    overlaps_found = any(
        chunks[i + 1].char_start < chunks[i].char_end for i in range(len(chunks) - 1)
    )
    assert overlaps_found


def test_oversized_single_paragraph_is_split_into_token_windows():
    paragraph = " ".join(["The quick brown fox jumps over the lazy dog."] * 200)

    chunks = chunk_text_blocks([paragraph], FILING_ID, "1A", "Risk Factors", 0)

    assert len(chunks) > 1
    assert all(c.token_count <= settings.max_tokens for c in chunks)
    for chunk in chunks:
        assert paragraph[chunk.char_start : chunk.char_end] == chunk.text


def test_empty_input_returns_no_chunks():
    assert chunk_text_blocks([], FILING_ID, "1", "Business", 0) == []


def test_order_index_and_chunk_ids_are_sequential():
    paragraphs = ["Para one.", "Para two.", "Para three."]
    chunks = chunk_text_blocks(paragraphs, FILING_ID, "1", "Business", 5)
    order_indices = [c.order_index for c in chunks]
    assert order_indices == sorted(order_indices)
    assert order_indices[0] >= 5
    assert all(c.chunk_id.startswith(f"{FILING_ID}:1:") for c in chunks)
