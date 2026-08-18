# ADASPilot

## What this is

Finding reliable evidence across ADAS papers and safety documents during a design
review is slow, especially when the answer depends on a table, diagram, or more
than one source. I built ADASPilot to answer those questions over a verified PDF
corpus with exact page and figure citations—or abstain when the available evidence
is not good enough.

The project is deliberately backend-first. Its main value is the retrieval,
agent-control, safety, and evaluation work; the FastAPI and Streamlit layer is only
a small interface for demonstrating that work.

## What I found while building it

**HyDE reduced retrieval quality on this corpus.** HyDE is often recommended for
advanced RAG, so I implemented it as a controlled experiment. On 63 answerable
development questions, adding conditional HyDE reduced Recall@5 from 0.873 to
0.810. ADAS terminology is unusually precise—terms such as 77 GHz, ESC, yaw rate,
and specific protocol names already make strong queries. My interpretation is
that the hypothetical passage introduced vocabulary drift. HyDE remains available
for category-level experiments but is disabled in the default pipeline.

**BM25 mattered more than I expected.** The largest single gain came from moving
from dense retrieval (Recall@5 0.698) to dense plus BM25/RRF (0.794). Exact
identifiers and acronyms are common in automotive documents, and sparse retrieval
preserved them better than embeddings alone.

**A valid citation is not the same as good citation recall.** Plan mode already
rejected citations outside its evidence board, but its first citations could all
come from one parallel research task. I changed synthesis to put the strongest
evidence item from each task first, then add supporting context. The verifier still
accepts only exact document, page, record, and asset IDs from the shared evidence
board. On the complete development rerun, plan citation recall increased from
0.222 to 0.556 and text Recall@5 increased from 0.625 to 0.875.

**Safety-critical web fallback should be narrow.** ReAct mode can optionally use a
Tavily literature search, but only after local corpus retrieval returns no evidence.
It is restricted to arXiv, IEEE, SAE, NHTSA, Euro NCAP, and ResearchGate; results
are labelled as web evidence and receive a conservative support score. With no API
key, the tool is not registered.

## What it does

- Ingests text, tables, page images, figures, captions, OCR tokens, and bounding
  boxes from versioned PDFs.
- Retrieves with dense search, BM25, reciprocal-rank fusion, cross-encoder
  reranking, parent context, and an optional visual index.
- Runs three bounded modes: direct corrective retrieval, ReAct research, and
  plan-and-execute for multi-part questions.
- Applies PII redaction, prompt-injection checks, access filters, tool permissions,
  image limits, cycle limits, and exact citation validation.
- Evaluates retrieval, citation, visual, abstention, routing, tool, plan, latency,
  token, and cost behavior separately for each execution mode.
- Exposes document management, streamed answers, evidence, IEEE-style references,
  figure crops, timelines, and evaluation results through a small HTTP-only UI.

## System flow

```text
Verified PDFs
  -> page/table/figure/OCR extraction
  -> parent and child chunks + visual records
  -> Qdrant dense index + BM25 + reranker + visual index
  -> direct | ReAct | plan-and-execute
  -> guardrails + evidence-board citation verification
  -> FastAPI -> minimal Streamlit UI
  -> mode-separated evaluation report
```

## Measured results

### Text retrieval ablation

| Experiment | Recall@1 | Recall@5 | Recall@10 | MRR | Decision |
|---|---:|---:|---:|---:|---|
| E0 dense | 0.333 | 0.698 | 0.921 | 0.490 | baseline |
| E1 dense + BM25/RRF | 0.429 | 0.794 | 0.905 | 0.578 | keep |
| E2 + cross-encoder | 0.476 | 0.857 | 0.937 | 0.631 | keep |
| E3 + parent context | 0.476 | 0.873 | 0.937 | 0.642 | selected |
| E4 + conditional HyDE | 0.444 | 0.810 | 0.905 | 0.592 | disabled |

### End-to-end development baseline

| Mode | Examples | Recall@5 | MRR | Citation recall | Goal accuracy | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 61 | 0.882 | 0.601 | 0.727 | 0.787 | 1,475 ms |
| ReAct | 10 | 0.900 | 0.650 | 0.583 | 1.000 | 1,372 ms |
| Plan | 9 | 0.875 | 0.583 | 0.556 | 1.000 | 2,782 ms |

These are deterministic local results, not resume placeholders. The run used no
external model calls, so recorded API cost was zero. The CLIP visual baseline is
Recall@1 0.20 and Recall@5 0.40 over 10 development visual questions.

## Known limitations

- Visual Recall@5 is 0.40; better caption/OCR fusion and the optional ColPali
  backend still need controlled comparison.
- Direct-mode abstention F1 is 0.381; the evidence threshold needs calibration.
- Plan citation recall improved, but citation precision is still 0.25; evidence
  selection needs further calibration before treating plan mode as final.
- RAGAS faithfulness needs an external judge model. Deterministic metrics remain
  the default and unavailable semantic scores stay null.
- The committed benchmark contains five verified public documents. Local uploads,
  copyrighted PDFs, generated indexes, and private manifest entries remain ignored.
- Upload processing runs as a visible background job and reuses unchanged extraction
  and visual embeddings, but rebuilding a large combined text index is still a
  laptop-scale operation rather than a distributed document service.
- Web results are clearly separated from the verified corpus and are never treated
  as equivalent evidence.

## Setup

Requirements: Python 3.11 or 3.12, Make, and Tesseract for OCR.

```bash
cp .env.example .env
make setup
make verify

make ingest
make build-index
make experiments
make build-visual-index
make evaluate-visual
make evaluate-system
```

The first index commands may download the configured embedding, reranking, and
visual models. Later runs use the local cache.

Run the three workflows directly:

```bash
make orchestrate QUERY="What does automotive radar measure?" MODE=direct
make orchestrate QUERY="Research radar measurement evidence." MODE=react
make orchestrate QUERY="Compare radar and camera strengths." MODE=plan
```

Run the local application in separate terminals:

```bash
make install-ui
make run-api
make run-ui
```

The API documentation is available at `http://127.0.0.1:8000/docs`, and the UI
normally opens at `http://localhost:8501`.

### Optional bounded web fallback

```bash
make install-web
```

Then set `ADASPILOT_TAVILY_API_KEY` in `.env`. The tool remains ReAct-only and is
called only after an empty local-corpus search.

## Corpus management

The Streamlit sidebar communicates only with FastAPI:

- `POST /corpus/upload` validates and stages a PDF.
- `POST /corpus/ingest-staged` starts a bounded background ingestion and index job.
- `GET /corpus/ingest-status` reports validation, extraction, text-index, visual-index,
  completion, or failure state without holding a long browser request open.
- `DELETE /corpus/documents/{document_id}` marks a document inactive without
  deleting its source PDF.
- `GET /corpus/documents/{document_id}/citation` formats stored bibliographic
  metadata as an IEEE-style reference.

## Project structure

```text
src/adaspilot/
├── api/             # corpus, query/SSE, evaluation, and asset endpoints
├── cli/             # small command-line entry points
├── config/          # typed settings
├── domain/          # manifests, evidence, citations, answers, and metrics
├── evaluation/      # deterministic and optional RAGAS evaluation
├── ingestion/       # PDF extraction, chunking, and versioned builds
├── orchestration/   # direct, ReAct, plan-and-execute, and local tools
├── retrieval/       # embeddings, Qdrant, E0-E4, and visual retrieval
├── safety/          # input, access, tool, image, plan, and citation checks
└── tools/           # optional bounded external literature search

apps/streamlit_app.py # minimal HTTP-only UI
```

Implementation details are kept in
[`docs/modules_2_to_5.md`](docs/modules_2_to_5.md),
[`docs/modules_6_to_9.md`](docs/modules_6_to_9.md), and
[`docs/modules_10_to_11.md`](docs/modules_10_to_11.md).
