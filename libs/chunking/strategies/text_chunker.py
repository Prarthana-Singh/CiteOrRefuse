"""Packs a section's paragraphs into overlapping, token-bounded chunks."""
from libs.core.config import settings
from libs.core.models import Chunk, ChunkType
from libs.chunking.token_utils import count_tokens, decode, encode, split_into_token_windows


def chunk_text_blocks(
    paragraphs: list[str],
    filing_id: str,
    section_item_code: str,
    section_title: str,
    start_order_index: int,
    part: str | None = None,
) -> list[Chunk]:
    """Packs whole paragraphs into ~target_tokens chunks.

    Paragraphs are never split mid-sentence unless a single paragraph alone
    exceeds `max_tokens`, in which case that paragraph is split into token
    windows on its own. Consecutive chunks overlap by re-including whichever
    trailing paragraphs of the previous chunk fit within the overlap budget.
    """
    if not paragraphs:
        return []

    target = settings.target_tokens
    max_tokens = settings.max_tokens
    overlap_budget = settings.overlap_tokens

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for para in paragraphs:
        start = cursor
        end = start + len(para)
        offsets.append((start, end))
        cursor = end + 2  # "\n\n" join separator

    para_tokens = [count_tokens(p) for p in paragraphs]

    chunks: list[Chunk] = []
    order_index = start_order_index
    n = len(paragraphs)
    i = 0

    while i < n:
        if para_tokens[i] > max_tokens:
            windows = _split_oversized_paragraph(
                paragraphs[i],
                offsets[i][0],
                filing_id,
                section_item_code,
                section_title,
                order_index,
                target,
                overlap_budget,
                part,
            )
            chunks.extend(windows)
            order_index += len(windows)
            i += 1
            continue

        indices = [i]
        total = para_tokens[i]
        j = i + 1
        while j < n and total + para_tokens[j] <= max_tokens and total < target:
            indices.append(j)
            total += para_tokens[j]
            j += 1

        text = "\n\n".join(paragraphs[idx] for idx in indices)
        chunks.append(
            Chunk(
                chunk_id=f"{filing_id}:{section_item_code}:{order_index}",
                filing_id=filing_id,
                chunk_type=ChunkType.TEXT,
                section_item_code=section_item_code,
                section_title=section_title,
                part=part,
                order_index=order_index,
                char_start=offsets[indices[0]][0],
                char_end=offsets[indices[-1]][1],
                token_count=count_tokens(text),
                text=text,
            )
        )
        order_index += 1

        overlap_count = 0
        overlap_tok = 0
        for idx in reversed(indices):
            if overlap_tok + para_tokens[idx] > overlap_budget:
                break
            overlap_tok += para_tokens[idx]
            overlap_count += 1

        # Guarantee forward progress even if the whole chunk fits the overlap budget.
        i = max(j - overlap_count, i + 1) if overlap_count and j < n else j

    return chunks


def _split_oversized_paragraph(
    paragraph: str,
    char_offset: int,
    filing_id: str,
    section_item_code: str,
    section_title: str,
    start_order_index: int,
    target: int,
    overlap: int,
    part: str | None = None,
) -> list[Chunk]:
    tokens = encode(paragraph)
    order_index = start_order_index
    result: list[Chunk] = []
    for start_idx, window in split_into_token_windows(tokens, target, overlap):
        window_text = decode(window)
        char_start = char_offset + len(decode(tokens[:start_idx]))
        result.append(
            Chunk(
                chunk_id=f"{filing_id}:{section_item_code}:{order_index}",
                filing_id=filing_id,
                chunk_type=ChunkType.TEXT,
                section_item_code=section_item_code,
                section_title=section_title,
                part=part,
                order_index=order_index,
                char_start=char_start,
                char_end=char_start + len(window_text),
                token_count=count_tokens(window_text),
                text=window_text,
            )
        )
        order_index += 1
    return result
