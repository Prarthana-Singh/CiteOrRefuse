# CiteOrRefuse

A retrieval-augmented Q&A system over SEC filings (10-Ks, 10-Qs) built
around one contract: **every answer is either fully cited to a specific
retrieved passage, or the system explicitly refuses rather than guess.**

## What this is

Ask a question about a filing and you get one of two things back: a
grounded answer, where every factual claim is attributed to a specific
retrieved chunk you can go read yourself, or an explicit refusal — "I don't
have sufficient evidence in the retrieved filings to answer this
confidently" — when the evidence isn't actually there. There is no third
option where the system answers anyway and hopes it's right. This matters
specifically for financial filings because the failure mode of a
plausible-sounding wrong number isn't a minor inconvenience — SEC filings
are the legal record investors, analysts, and regulators rely on, and a
hallucinated figure that reads exactly like a real one is more dangerous
than an obvious error. A tool that gets this wrong quietly is worse than a
tool that admits when it doesn't know.

## Architecture

Full walkthrough with a diagram: [docs/architecture.md](docs/architecture.md).

Five layers, built and tested in order, each hardened before the next was
built on top of it:

1. **Ingestion** (`libs/ingestion`, `libs/chunking`) — parses real EDGAR
   HTML into ordered blocks, detects `Item`/`Part` structure, chunks text
   and tables (never splitting a table row) into token-bounded, overlapping
   chunks.
2. **Embedding + Indexing** (`libs/embedding`, `libs/vectorstore`,
   `libs/indexing`) — dense (OpenAI) and sparse (BM25 via `fastembed`)
   vectors per chunk in one Qdrant collection, with citation-complete
   payloads and idempotent upserts.
3. **Retrieval** (`libs/retrieval`) — Qdrant-native reciprocal rank fusion
   of dense + sparse, then Cohere Rerank.
4. **Generation + Groundedness Gate** (`libs/generation`) — structured,
   per-claim-cited generation, verified claim-by-claim by an LLM-as-judge
   before anything is returned.
5. **Eval / CI** (`libs/eval`, `.github/workflows/eval.yml`) — a
   hand-verified Q&A regression suite run against the real pipeline, gating
   on exit code.
6. **Serving** (`libs/api`) — a FastAPI wrapper: `POST /answer` (the same
   pipeline every script above already calls) plus `POST /ingest`, a
   real user-facing upload path onto Phase 1/2's *unchanged* ingestion and
   indexing functions. A minimal Streamlit app (`streamlit_app.py`) is a
   thin client over both endpoints — see Limitations for what this layer
   does and doesn't do yet.

## Why this is harder than a typical RAG tutorial

Three failure modes drove almost every design decision here, not as an
afterthought but as the actual reason each layer looks the way it does:

- **Bad chunking loses structure.** Real EDGAR filings lay out headings
  inside single-cell tables for pixel alignment, split words across
  adjacent `<span>` tags, and paginate long sections with repeated running
  headers. A naive parser either treats headings as opaque table data
  (missing every section boundary) or fragments one section into dozens of
  near-empty ones. Tables get chunked row-group-wise specifically so a
  number is never separated from the row label that gives it meaning.
- **Incomplete retrieval misses cross-references.** A single fact — a
  segment's net sales — often appears in more than one place (an MD&A
  table and a financial-statement note) at different levels of detail.
  Retrieval has to surface enough of the *right* candidates for generation
  to actually find the complete picture, not just the first mention.
- **Hallucination is the default failure mode of every other design
  choice going right.** Good chunking and good retrieval still leave an
  LLM free to state something plausible but false, or to cite a real
  source that doesn't actually say what the claim says it does. The
  groundedness gate exists because the first two layers working correctly
  is necessary but not sufficient.

## Real bugs found

All found running the pipeline against real filings (Amazon, Tesla,
Microsoft 10-Ks; an Apple 10-Q), not synthetic test fixtures — the
project's only synthetic fixture didn't catch any of these.

| Bug | Symptom |
|---|---|
| Table-wrapped headings | Real filings lay out *every* heading in a single-row, single-cell `<table>`; the parser treated all tables as opaque data, so zero heading text ever reached section detection |
| Silent byte-truncation | Pre-decoding filing bytes to a Python `str` before handing them to lxml silently truncated the parse tree partway through a ~2MB document — no exception, 329 blocks instead of 856 |
| Mid-word text corruption | `get_text(separator=" ")` inserted a space at every inline-tag boundary, including where a filing's generator split a single word (`STATE`/`MENTS`) across two adjacent `<span>`s with no space in the source |
| Running-header fragmentation | Repeated `PART II`/`Item 8` pagination headers on every printed page caused one section to fragment into ~40 near-empty sections with junk titles |
| Loose `PART` regex + fragile dedup key | `_PART_RE` only matched a bare `PART I` with nothing after it, so a filer writing `PART I – FINANCIAL INFORMATION` got `part=None` everywhere; a related fix (deduping repeated running headers by item code alone) was fragile for 10-Qs, which legitimately reuse Item codes 1-4 across Part I and Part II |
| Citation-binding non-determinism | Generation occasionally (5% of runs, confirmed over 20 repeated identical runs) attributed a factually-correct claim to the wrong retrieved chunk when two chunks discussed the same topic at different levels of detail — see below |

## The eval harness: proof, not just a report

Most RAG portfolio projects don't have automated regression testing for
retrieval or generation quality. This one does, and the harness has
actually caught something, not just run clean.

**Proof the CI gate works**: `scripts/run_eval.py` exits non-zero on any
case failure. To confirm that's real and not just a script that happens to
always exit 0, the "refuse if unsupported" instruction was deliberately
removed from the generation prompt — a real, understood regression, not a
synthetic test hook — and the harness was run before, after, and after
reverting: **4/4 passed (exit 0) → 0/4 passed (exit 1, every case
correctly diagnosed as "expected refused, got answered") → reverted,
confirmed byte-identical via `git diff` → 4/4 passed (exit 0) again.**

**A real bug the eval harness surfaced, not a planted one**: a case asking
Tesla's reportable business segments — a fact stated in one unambiguous
sentence — flipped between "answered confidently" and "refused" across
identical runs of the same query against the same index. Investigating it
properly (20 repeated runs, logging every stage) showed retrieval was
100% deterministic and claim *content* was stable word-for-word, but
generation occasionally bound a correct claim to the wrong chunk_id: Tesla's
detailed segment description lives in a chunk that also contains an
unrelated related-party-transactions note (packed together by the chunker's
token-budget logic), and one retrieved chunk mentions segments only in
passing. 19 times out of 20, the claim was correctly attributed to the
detailed chunk; once, it was attributed to the terse one instead, and the
groundedness gate correctly refused rather than let the misattribution
through. The fix — citation-rebind — checks other already-retrieved chunks
before refusing a claim that fails its cited source, and re-attributes it
if one actually supports it, never inventing a new source. Re-running the
same 20x investigation post-fix: **20/20 answered, 0/20 refused**, with the
exact same misattribution recurring live on one run and being caught and
corrected instead of sinking the answer. Implementing the fix also
surfaced a second, downstream bug: citations were being built from the
generator's original (pre-correction) claim list, which meant a corrected
verdict wasn't reflected in what the user would actually see — fixed
alongside the rebind logic itself.

## Current status and honest limitations

- **CI is manually triggered, not automatic.** `.github/workflows/eval.yml`
  is `workflow_dispatch`-only. It doesn't run on every PR, both because
  each run costs real OpenAI/Cohere API usage and because repo secrets
  would need to be available on every triggering context. This was a
  deliberate "start conservative" choice, not an oversight — worth
  revisiting once the eval set is larger and scores have proven stable
  over time.
- **The eval set is 17 hand-verified cases, not hundreds.** Cases were
  written by reading each filing directly and committing to an expected
  outcome *before* running the pipeline, specifically to avoid an eval set
  that only proves the system can replay behavior already known to be
  correct — but 17 cases across 4 filings is a start, not real coverage.
- **Citation-rebind has a known, documented, unsolved gap.** It only
  fires when a claim's originally-cited chunk *fails* its groundedness
  check. A claim mis-bound to a chunk that only loosely or imprecisely
  supports it — close enough to pass the check — never reaches the rebind
  path and is invisible to both the fix and the eval harness's current
  metrics. This is recorded in `libs/generation/groundedness.py`'s module
  docstring and in the affected eval case's notes, not silently left
  implicit.
- **The API layer is a demo service, not a production one.** The 4 fixture
  filings are indexed into an in-memory Qdrant collection lazily on the
  first request; `POST /ingest` adds real uploads to that same in-memory
  collection, but nothing is persisted to disk, so a process restart loses
  everything (fixtures re-index automatically; uploads don't). There's no
  auth, and `/answer` itself has no rate limiting or request queuing (only
  `/ingest` does, being the expensive/abusable operation) — it inherits
  whatever latency and rate limits OpenAI/Cohere impose underneath.
- **Ingestion is HTML-only and session-scoped, by design, not as a gap.**
  `POST /ingest` accepts `.htm`/`.html` 10-Ks only (PDF support is a
  different, larger project — a different parser, different structure
  assumptions throughout ingestion/chunking). An uploaded filing is
  retrievable only by a caller who passes back its `filing_id`; an
  unscoped `/answer` query always stays pinned to the 4 baked-in fixtures,
  never picking up someone else's upload, and nothing about the uploaded
  filing is deleted or cleaned up afterward — the simplest safe option at
  this project's scale, not a real multi-tenant access-control model.
- **The existing `UnsupportedDocumentTypeError` guard is the only content
  safety net**, now doing real work against arbitrary user uploads instead
  of just internal fixtures — verified live by uploading a real 10-Q
  through the running endpoint and confirming a clean, specific 422
  rather than a crash or a plausible-looking wrong answer.
- **The Streamlit frontend is manually verified only.** It's a thin
  display layer over `/ingest` and `/answer` with no business logic of its
  own to unit test; `tests/api` already covers everything it calls.
- **140 tests pass** (`pytest tests/`), none requiring a live API key or a
  running Qdrant instance — every layer, including the API layer, is
  independently testable with injected fakes/dependency overrides. Live
  verification against real OpenAI/Cohere/Qdrant has been run manually at
  each phase, including uploading a real, never-before-used 10-K (NVIDIA,
  fetched fresh from EDGAR) through the running API and getting a
  correctly grounded, cited answer back on genuinely new input.
