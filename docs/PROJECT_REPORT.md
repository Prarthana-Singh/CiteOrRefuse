# CiteOrRefuse — Project Report

## What I built

I built a retrieval-augmented Q&A system over SEC filings (10-Ks, 10-Qs)
around one contract: every answer is either fully cited to a specific
retrieved passage, or the system explicitly refuses rather than guess.
Ask it a question about a filing and you get one of two things back — a
grounded answer where every factual claim is attributed to a chunk of
the real filing you can go read yourself, or an explicit refusal ("I
don't have sufficient evidence in the retrieved filings to answer this
confidently"). There is no third option where it answers anyway and
hopes it's right.

I built it this way because of what the domain actually punishes. SEC
filings are the legal record investors, analysts, and regulators rely
on. A hallucinated number that reads exactly like a real one is more
dangerous than an obvious error — a tool that gets this wrong quietly is
worse than a tool that admits when it doesn't know. That's the whole
premise of the project, not a marketing line: I designed every layer
around making the refusal path as trustworthy as the answer path.

## Why I think this is harder than a typical RAG tutorial

I let three failure modes drive almost every design decision, not as an
afterthought but as the actual reason each layer looks the way it does:

- **Bad chunking loses structure.** Real EDGAR filings lay out headings
  inside single-cell tables for pixel alignment, split words across
  adjacent `<span>` tags, and paginate long sections with repeated
  running headers. A naive parser either treats headings as opaque table
  data (missing every section boundary) or fragments one section into
  dozens of near-empty ones. I chunk tables row-group-wise specifically
  so a number is never separated from the row label that gives it
  meaning.
- **Incomplete retrieval misses cross-references.** A single fact — a
  segment's net sales — often appears in more than one place (an MD&A
  table and a financial-statement note) at different levels of detail.
  Retrieval has to surface enough of the *right* candidates for
  generation to actually find the complete picture, not just the first
  mention.
- **Hallucination is the default failure mode of every other design
  choice going right.** Good chunking and good retrieval still leave an
  LLM free to state something plausible but false, or to cite a real
  source that doesn't actually say what the claim says it does. I built
  the groundedness gate because the first two layers working correctly
  is necessary but not sufficient.

## Architecture — six layers

Full diagram: [docs/architecture.md](architecture.md). In build order:

1. **Ingestion** (`libs/ingestion`, `libs/chunking`) — parses real EDGAR
   HTML into ordered blocks, detects `Item`/`Part` structure, chunks
   text and tables (never splitting a table row) into token-bounded,
   overlapping chunks.
2. **Embedding + Indexing** (`libs/embedding`, `libs/vectorstore`,
   `libs/indexing`) — dense (OpenAI `text-embedding-3-small`) and sparse
   (BM25 via `fastembed`) vectors per chunk in one Qdrant collection,
   with citation-complete payloads and idempotent upserts.
3. **Retrieval** (`libs/retrieval`) — Qdrant-native reciprocal rank
   fusion of dense + sparse, then Cohere Rerank.
4. **Generation + Groundedness Gate** (`libs/generation`) — structured,
   per-claim-cited generation, verified claim-by-claim by an
   LLM-as-judge before anything is returned. Includes citation-rebind
   (below).
5. **Eval / CI** (`libs/eval`, `.github/workflows/eval.yml`) — a
   hand-verified Q&A regression suite run against the real pipeline,
   gating on exit code.
6. **Serving** (`libs/api`) — a FastAPI wrapper: `POST /answer` and
   `POST /ingest`, a real user-facing upload path.

**Tech stack**: Python 3.11+, OpenAI (embeddings + generation + judge),
Qdrant (hybrid dense/sparse vector store), `fastembed` (local BM25),
Cohere Rerank, FastAPI, Pydantic throughout, pytest. No LlamaIndex — I
wrote the chunking/retrieval logic myself so I could shape it around
this domain's specific failure modes instead of a generic pattern.

## Real bugs I found and fixed

Every one of these came from running the pipeline against real filings
(Amazon, Tesla, Microsoft 10-Ks; an Apple 10-Q; later NVIDIA), not
synthetic test fixtures — my one synthetic fixture never caught any of
these:

| Bug | Symptom |
|---|---|
| Table-wrapped headings | Real filings lay out *every* heading in a single-row, single-cell `<table>`; my parser treated all tables as opaque data, so zero heading text ever reached section detection |
| Silent byte-truncation | Pre-decoding filing bytes to a Python `str` before handing them to lxml silently truncated the parse tree partway through a ~2MB document — no exception, 329 blocks instead of 856 |
| Mid-word text corruption | `get_text(separator=" ")` inserted a space at every inline-tag boundary, including where a filing's generator split a single word (`STATE`/`MENTS`) across two adjacent `<span>`s with no space in the source |
| Running-header fragmentation | Repeated `PART II`/`Item 8` pagination headers on every printed page caused one section to fragment into ~40 near-empty sections with junk titles |
| Loose `PART` regex + fragile dedup key | My `_PART_RE` only matched a bare `PART I` with nothing after it, so a filer writing `PART I – FINANCIAL INFORMATION` got `part=None` everywhere; a related fix (deduping repeated running headers by item code alone) was fragile for 10-Qs, which legitimately reuse Item codes 1–4 across Part I and Part II |
| Citation-binding non-determinism | Generation occasionally (5% of runs, confirmed over 20 repeated identical runs) attributed a factually-correct claim to the wrong retrieved chunk when two chunks discussed the same topic at different levels of detail — full story below |
| `/ingest` swallowing indexing failures | I only caught ingestion/chunking errors in the upload endpoint; a failure in the embedding/indexing step right after (a real OpenAI timeout or rate limit) leaked straight through as a bare, undetailed 500 — fixed with a proper try/except, server-side logging, and a specific 502 |
| Eval harness doubling Cohere calls | `run_eval` called `retrieve()` a second time per case just to get chunk_ids for scoring, silently doubling real Cohere rerank usage and causing rate-limit failures against the trial tier — fixed by reusing `answer()`'s own `retrieved_chunk_ids` |

## The citation-rebind story (the one I'd tell in detail)

While hardening the eval harness, one case — asking Tesla's reportable
business segments, a fact stated in one unambiguous sentence — flipped
between "answered confidently" and "refused" across identical runs of
the same query against the same index. I didn't accept "flaky LLM" as
an explanation. I ran the same query 20 times, logging every stage
(retrieval, raw claims, groundedness verdicts) and found retrieval was
100% deterministic and claim *content* was stable word-for-word —
generation occasionally bound a correct claim to the wrong `chunk_id`.
Tesla's detailed segment description lives in a chunk that also
contains an unrelated related-party-transactions note (packed together
by my chunker's token-budget logic), and one retrieved chunk mentions
segments only in passing. 19 times out of 20, the claim was correctly
attributed to the detailed chunk; once, it was attributed to the terse
one instead, and my groundedness gate correctly refused rather than let
the misattribution through.

I fixed it with **citation-rebind**: before refusing a claim that fails
its cited source, I check the *other* already-retrieved chunks for one
that actually supports it, and re-attribute if so — never inventing a
new source, only correcting which already-retrieved one a claim points
to. Re-running the same 20x investigation post-fix: **20/20 answered,
0/20 refused**, with the exact same misattribution recurring live on one
run and being caught and corrected instead of sinking the answer.
Implementing the fix also surfaced a second, downstream bug: I was
building citations from the generator's original (pre-correction) claim
list, so a corrected verdict wasn't reflected in what a user would
actually see — fixed alongside the rebind logic.

**Known, documented, unsolved gap**: rebind only fires when a claim's
originally-cited chunk *fails* its groundedness check. A claim mis-bound
to a chunk that only loosely or imprecisely supports it — close enough
to pass — never reaches the rebind path and is invisible to both the
fix and my eval harness's current metrics. I recorded this in
`libs/generation/groundedness.py`'s module docstring, not left silently
implicit.

## The eval harness: proof, not just a report

Most RAG portfolio projects don't have automated regression testing for
retrieval or generation quality. I built one, and it's actually caught
something, not just run clean.

**Proof the CI gate works**: `scripts/run_eval.py` exits non-zero on any
case failure. To confirm that's real and not just a script that happens
to always exit 0, I deliberately removed the "refuse if unsupported"
instruction from the generation prompt — a real, understood regression,
not a synthetic test hook — and ran the harness before, after, and after
reverting: **4/4 passed (exit 0) → 0/4 passed (exit 1, every case
correctly diagnosed as "expected refused, got answered") → reverted,
confirmed byte-identical via `git diff` → 4/4 passed (exit 0) again.**

**Current scale**: 17 hand-verified cases across 4 real filings (3
10-Ks + 1 10-Q). I wrote each case by reading the filing directly and
committing to an expected outcome *before* running the pipeline,
specifically to avoid an eval set that only proves the system can
replay behavior already known to be correct.

## The serving layer and the upload feature

`libs/api` is a FastAPI service with three endpoints:

- `GET /health`
- `POST /answer` — a query (optional `top_k`, optional `filing_id`)
  against either the 4 baked-in fixture filings (default) or a specific
  uploaded filing
- `POST /ingest` — a real, user-facing upload path: `.htm`/`.html` 10-Ks
  only (no PDF — a different, larger project with different parser
  assumptions throughout). I run every upload through my existing,
  unchanged ingestion + chunking + indexing pipeline, so the existing
  `UnsupportedDocumentTypeError` guard does real work here — a non-10-K
  upload (I tested with a real 10-Q) gets a clean, specific 422, not a
  crash or a plausible-looking wrong answer. I capped upload size and
  rate-limited `/ingest` specifically (not `/answer`), since ingestion
  is the expensive, abusable operation.

An uploaded filing is session-scoped by construction: it's retrievable
only by a caller who passes back the `filing_id` `/ingest` returned; an
unscoped `/answer` query always stays pinned to the 4 fixtures.

I built a Streamlit frontend alongside this, then deliberately removed
it to keep the project FastAPI-only — `streamlit_app.py` and its
dependencies are gone; `/docs` (Swagger UI), curl, or direct Python
calls are the only interfaces now.

## Live verification — including what I got wrong the first time

I didn't just run tests — I ran the real pipeline against real filings
with real API keys at every phase, and I held myself to actually
checking the results, not skimming them.

The most important instance of this: after building the upload feature,
I first verified it with a single query against a freshly-downloaded,
never-before-used NVIDIA 10-K, got a correct answer, and reported it as
"live-verified." That was true as far as it went, but it was one query,
and I was pushed to go back and actually scrutinize it the way I'd
scrutinized everything else in this project. When I did:

- **Ingestion structure held up.** All 23 sections detected correctly,
  every `part` populated. One thing initially looked like a bug — Item 8
  ("Financial Statements") was only 1 block while Item 15 had 397 — but
  I traced it against the real filing text and confirmed NVIDIA's own
  Item 8 explicitly points to Item 15, where the real financial
  statements and auditor's report genuinely live. Correct, not a bug.
- **A real retrieval-precision miss surfaced.** A query about NVIDIA's
  total operating expenses refused, even though the real figure
  ($23,076M) exists in a clean MD&A table. I traced why: that chunk was
  retrieved and ranked respectably (7th of 20, post-rerank) but fell
  outside the production default `top_k=5`, edged out by less-relevant
  footnote chunks. The system failed safe (refused instead of guessing)
  but this is a genuine, avoidable miss — a live instance of the
  "incomplete retrieval" failure mode I named above, not a new category.
- **A citation-precision issue surfaced.** A competition-risk-factors
  citation was factually correct (I confirmed the actual supporting text
  exists in the cited chunk) but that one chunk packs three unrelated
  risk sub-topics together, so its preview text looks irrelevant on a
  skim. Same root cause as the Tesla bug above, a different instance.

I'm reporting both as real, deferred findings — not fixed yet, not
hidden. I don't want to retune retrieval parameters off one test case
without checking it against the full 17-case eval suite first.

## An engineering incident worth including

Partway through this project, my local git branch and the pushed remote
branch diverged — a local reset had rewound past work that was already
pushed. I ended up mid-merge with unresolved conflicts across 7 files.
I diagnosed it properly via `git reflog` before touching anything,
confirmed every conflict was purely additive (nothing genuinely
contradictory), resolved by keeping the fuller side, verified the full
test suite still passed, and completed the merge. I'm including this
because it's a real thing that happened, not because it reflects well —
consistent with how I've treated every other finding in this project.

## Deliberate decisions I made, not gaps I overlooked

- **Postgres and S3, consciously skipped.** My original spec named both.
  Qdrant's payloads already carry every citation-relevant field a
  Postgres metadata store would have served — no second system to keep
  in sync. S3 would only matter for persistence across restarts, which
  was never a goal at this scale (everything is deliberately
  ephemeral/in-memory). The one thing that would justify Postgres — a
  real query-log/audit-trail feature — doesn't exist anywhere in the
  project today, so there's nothing currently pulling it back in.
- **HTML only, no PDF.** A different parser, different structural
  assumptions throughout ingestion and chunking — a different, larger
  project.
- **No authentication, no persistent storage, no Docker/K8s.**
  Deliberately deferred as follow-on infrastructure work, not attempted
  and abandoned.
- **CI is manually triggered (`workflow_dispatch`), not automatic.**
  Both because each run costs real OpenAI/Cohere usage and because repo
  secrets would need to be available on every triggering context — a
  "start conservative" choice I'd revisit once the eval set is larger.

## Current numbers

- **140 tests pass** (`pytest tests/`), none requiring a live API key or
  a running Qdrant instance — every layer, including the API layer, is
  independently testable with injected fakes/dependency overrides.
- **17 eval cases** across 4 real filings, run through the real
  pipeline, gating CI on exit code.
- **22 commits** of real feature work.
- **8 real bugs** found and fixed against real filings (5 in ingestion,
  1 in generation/citation-binding, 2 in the serving layer), none from
  my synthetic test fixture.
- Live-verified against 5 real companies' filings total (Amazon, Tesla,
  Microsoft, Apple, NVIDIA) with real API keys, not just unit tests.

## What I'd say if someone asked "is this finished?"

The core idea is fully built, tested, and honestly verified: ingest a
real filing, retrieve the right passages, generate a cited answer,
verify it's actually grounded, refuse when it isn't — with a CI harness
that's proven it catches real regressions, not just a report that
happens to run clean. What's left is infrastructure, not functionality:
containerizing it, persistent storage, authentication, automatic CI,
and the two retrieval-precision findings from the NVIDIA verification.
None of that changes whether the system works today — it does, on real
filings it had never seen before, including the times I checked closely
enough to find where it doesn't work perfectly yet.
