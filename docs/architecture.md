# Architecture

CiteOrRefuse is five layers, each built and tested independently before the
next one was built on top of it. No layer assumes the one below it is
perfect — the ingestion layer was hardened against four real filings before
embedding started, and the generation layer's groundedness gate exists
specifically because retrieval is expected to sometimes surface plausible
but wrong content, not because it's expected to be perfect.

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion (libs/ingestion, libs/chunking)"]
        A[Raw filing HTML] --> B[SecHtmlParser]
        B --> C[detect_sections]
        C --> D[chunk_filing_from_source]
    end

    subgraph Indexing["2. Embedding + Indexing (libs/embedding, libs/vectorstore, libs/indexing)"]
        D --> E[OpenAI dense embeddings]
        D --> F[fastembed BM25 sparse]
        E --> G[(Qdrant: one collection,<br/>dense + sparse vectors)]
        F --> G
    end

    subgraph Retrieval["3. Retrieval (libs/retrieval)"]
        Q[Query] --> H[embed_query: dense + sparse]
        H --> I[Qdrant native RRF fusion]
        G -.-> I
        I --> J[Cohere rerank]
    end

    subgraph Generation["4. Generation + Gate (libs/generation)"]
        J --> K[build_context]
        K --> L[generate_answer:<br/>structured claims + chunk_ids]
        L --> M{check_groundedness:<br/>per-claim LLM-as-judge}
        M -- claim fails --> N[rebind: try other<br/>retrieved chunks]
        N -- found --> M
        M -- all claims supported<br/>+ confidence >= threshold --> O[Answer, cited]
        M -- any claim fails,<br/>or below threshold --> P[Refuse]
    end

    subgraph EvalCI["5. Eval / CI (libs/eval, .github/workflows/eval.yml)"]
        R[EvalCase: query +<br/>expected behavior] --> S[run_eval:<br/>retrieve + answer]
        O --> S
        P --> S
        S --> T[score_case: refusal-correctness,<br/>retrieval recall, citation, content]
        T --> U[exit 0 / exit 1]
    end
```

## 1. Ingestion — `libs/ingestion`, `libs/chunking`

`SecHtmlParser` turns raw EDGAR HTML into an ordered list of `Block`s (text
or table), deliberately not trying to detect headings by tag — real filings
mark section headings with bold/underline styling, not semantic `<h1>`-`<h6>`
tags, or even lay them out inside single-row tables for pixel alignment.
`detect_sections` pattern-matches `Item N` headings across those blocks into
`Section`s with a `part` (I–IV). `chunk_filing_from_source` then packs each
section's text into ~400-600 token chunks via `tiktoken`, and chunks tables
row-group-wise, guaranteeing a chunk boundary never falls mid-row.

Every design choice here exists because of a specific real bug found
running the pipeline against a real filing — see the README's bug list.

## 2. Embedding + Indexing — `libs/embedding`, `libs/vectorstore`, `libs/indexing`

Each chunk is embedded twice: dense via OpenAI (`text-embedding-3-small`),
sparse via `fastembed`'s BM25 implementation (not hand-rolled — same
reasoning throughout this project: don't reimplement well-tested retrieval
math). Both vectors land in one Qdrant collection per point, not two
separate systems, with a deterministic point ID (`uuid5` of the chunk's own
`chunk_id`) so re-indexing an already-indexed filing overwrites rather than
duplicates. Every point's payload carries everything needed to reconstruct
a citation on its own — company, form type, filing date, Part, Item, section
title, char offsets, and the chunk's full text — so a retrieved point never
needs a second lookup to become a citation.

## 3. Retrieval — `libs/retrieval`

A query is embedded the same two ways (dense + sparse), but *without* the
context-header prefix chunks get at index time — a query is already
semantically complete; the header exists to anchor otherwise-bare chunk
content (like a table of numbers) topically, which a query doesn't need.
Qdrant's native reciprocal rank fusion (`prefetch` + `FusionQuery`) merges
the two branches — not a hand-rolled fusion implementation. The fused
candidates then go through Cohere Rerank, which reads actual content rather
than just rank position; see the README for a specific, measured example of
why that step earns its place in the pipeline rather than just re-sorting
what fusion already had right.

## 4. Generation + Groundedness Gate — `libs/generation`

Retrieved chunks are formatted into a context block tagged by `chunk_id`,
sent to the LLM with instructions to answer using *only* that context and
attribute every claim to a specific `chunk_id`. `check_groundedness` then
verifies each claim in two stages: first mechanically (is the cited
`chunk_id` even among what was retrieved — catches a hallucinated
reference with no LLM call needed), then semantically (does the cited
chunk's actual text support the claim, via a second LLM-as-judge call). If
a claim fails the semantic check, `check_groundedness` searches the *other*
already-retrieved chunks for one that does support it before giving up —
never inventing new sources, only correcting which already-retrieved one a
claim is attributed to. `overall_confidence` is the *minimum* across
claims, not an average: one unsupported claim sinks the whole answer.
Anything short of "all claims supported, confidence above threshold"
refuses, with a stated reason, rather than returning a partial or
hedged guess.

## 5. Eval / CI — `libs/eval`, `.github/workflows/eval.yml`

`EvalCase`s (query + expected behavior, optionally expected citations and
answer content) get run through the real `retrieve()` + `answer()`
pipeline; `score_case` checks four independent things — did it
answer/refuse correctly, was the expected chunk actually retrieved, was it
actually cited, did the answer contain the expected content — and
`scripts/run_eval.py` exits non-zero if any case fails. The GitHub Actions
workflow is `workflow_dispatch`-only (manually triggered, not on every
PR) — see the README for why, and for the current honest limitations of
this layer.
