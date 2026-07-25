"""Turns retrieval results into a prompt-ready, citable context block."""
from libs.chunking.token_utils import count_tokens
from libs.core.config import generation_settings
from libs.retrieval.pipeline import RetrievedResult


def _format_block(payload: dict) -> str:
    return (
        f"[chunk_id: {payload['chunk_id']}] "
        f"{payload['company']} {payload['form_type']}, "
        f"Part {payload['part']} Item {payload['section_item_code']} "
        f"({payload['section_title']}):\n{payload['text']}"
    )


def build_context(results: list[RetrievedResult]) -> str:
    """Dedupes exact-duplicate chunk text and enforces a token budget.

    Each block is tagged with its `chunk_id` -- the stable identifier the
    generator is instructed to cite claims by, rather than introducing a
    separate [1]/[2] numbering scheme on top of an identifier that already
    exists and is already what the groundedness gate looks candidates up
    by.

    Dedup here is exact-text only, a safety net against the same chunk
    being retrieved twice -- not fuzzy/near-duplicate detection. Retrieval
    routinely surfaces genuinely *different* chunks with overlapping
    content (e.g. the same figures appearing in an Item 7 MD&A table and
    an Item 8 financial-statement table), which is not what this guards
    against and is left to the generator/groundedness stages to handle.

    `results` is assumed already in descending relevance order (as
    `retrieve()` returns it), so when the token budget is exceeded, the
    lowest-relevance results are dropped first.
    """
    seen_texts: set[str] = set()
    blocks: list[str] = []
    budget = generation_settings.context_token_budget
    used_tokens = 0

    for result in results:
        text = result.payload["text"]
        if text in seen_texts:
            continue

        block = _format_block(result.payload)
        block_tokens = count_tokens(block)
        if blocks and used_tokens + block_tokens > budget:
            break

        seen_texts.add(text)
        blocks.append(block)
        used_tokens += block_tokens

    return "\n\n---\n\n".join(blocks)
