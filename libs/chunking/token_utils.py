"""Tokenizer-backed helpers for sizing and splitting chunk text.

tiktoken's BPE encoding is a lossless partition of the original UTF-8 bytes:
consecutive tokens' byte ranges are contiguous and non-overlapping, so
`decode(tokens[a:b])` always yields an exact substring of the text that was
encoded. That property is what lets the chunkers compute exact character
offsets for token-window splits without re-scanning the source text.
"""
import tiktoken

from libs.core.config import settings

_encoding = tiktoken.get_encoding(settings.tokenizer_encoding)


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def encode(text: str) -> list[int]:
    return _encoding.encode(text)


def decode(tokens: list[int]) -> str:
    return _encoding.decode(tokens)


def split_into_token_windows(
    tokens: list[int], target: int, overlap: int
) -> list[tuple[int, list[int]]]:
    """Splits a token list into overlapping windows of size <= target.

    Returns (start_index, window_tokens) pairs so callers can derive exact
    character offsets via `decode(tokens[:start_index])`.
    """
    if len(tokens) <= target:
        return [(0, tokens)]

    step = max(target - overlap, 1)
    windows: list[tuple[int, list[int]]] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + target]
        windows.append((start, window))
        if start + target >= len(tokens):
            break
        start += step
    return windows
