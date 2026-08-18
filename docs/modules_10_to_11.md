# Modules 10-11: Evaluation and Minimal Application

These modules add one evaluation contract and one small presentation boundary.
They do not introduce deployment infrastructure, background workers, Docker, a
second database, or frontend business logic.

## Module 10: Evaluation and LangSmith

Run the complete development benchmark:

```bash
make evaluate-system
```

For a quick smoke run:

```bash
.venv/bin/adaspilot-evaluate-system --split development --max-examples-per-mode 2
```

The report is saved to `data/results/system_evaluation.json`. Each example keeps
its retrieved pages, cited pages, expected/actual tools, terminal reason, and
latency so aggregate numbers can be audited.

### Why there are two metric groups

The default group is deterministic and does not need an API key:

- Text Recall@5, MRR, and nDCG@10 use verified document/page IDs.
- Visual Recall@5 uses verified asset IDs, with page evidence as a fallback.
- Context precision/recall compare retrieved page IDs with verified evidence.
- Citation precision/recall compare answer citations with verified evidence.
- Abstention F1 treats abstention as the positive class.
- Router accuracy compares the explainable router with the annotated mode.
- Tool precision/recall/F1 compare tool sets with required modalities.
- Plan validity, task completion, agent goal accuracy, and repeated-step rate
  come from typed workflow results.
- p50/p95 latency is wall-clock time. Token and cost values come from the model
  provider's `usage` response; deterministic local runs correctly report zero.

RAGAS context precision/recall, faithfulness, and multimodal faithfulness are
model-judged metrics. They remain `null` with status `not_configured` unless a
judge model is deliberately configured. This prevents local lexical proxies
from being mislabeled as RAGAS scores.

```bash
make install-evaluation

# Configure ADASPILOT_LLM_BASE_URL, ADASPILOT_LLM_API_KEY,
# ADASPILOT_LLM_MODEL, and ADASPILOT_RAGAS_JUDGE_MODEL before evaluating.
```

LangSmith is also opt-in. Local JSON reports are always written. Set
`ADASPILOT_LANGSMITH_TRACING=true` and `ADASPILOT_LANGSMITH_API_KEY` to send
evaluation traces to the configured project. This makes external data transfer
an explicit choice.

### Current measured development run

This is the full 80-example development split, using annotated benchmark modes,
the E3 retriever, deterministic policies, and extractive generation.

| Mode | Examples | Text Recall@5 | MRR | Citation recall | Goal accuracy | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 61 | 0.882 | 0.601 | 0.727 | 0.787 | 1,475 ms |
| ReAct | 10 | 0.900 | 0.650 | 0.583 | 1.000 | 1,372 ms |
| Plan | 9 | 0.875 | 0.583 | 0.556 | 1.000 | 2,782 ms |

The router matched all 80 annotated modes. ReAct tool-call F1 was 0.683; plan
tool-call F1 was 0.971; plan validity and task completion were both 1.0. Direct
abstention F1 remains 0.381. Task-balanced evidence selection raised plan
citation recall from 0.222 to 0.556, although plan citation precision remains
0.25. These are measured results, not resume placeholders. RAGAS metrics were
not run because no external judge was configured.

## Module 11: FastAPI and Streamlit

Start two local processes:

```bash
make install-ui
make run-api
make run-ui
```

The API provides:

- `GET /health`
- `GET /corpus/status`
- `POST /corpus/upload`
- `POST /corpus/ingest-staged`
- `DELETE /corpus/documents/{document_id}`
- `GET /corpus/documents/{document_id}/citation`
- `POST /query`
- `POST /query/stream`
- `GET /evaluation/latest`
- `GET /assets/{path}` for retrieved page and figure images

`/query/stream` emits small SSE events: accepted, execution, timeline, answer,
evidence, and done. The current local generator produces the answer first and
then streams it in display-sized chunks; it does not falsely claim provider
token streaming.

PDF uploads are first staged in `data/corpus/uploads`. Processing is a separate,
explicit action that validates each PDF again, updates the manifest, runs
versioned ingestion, and rebuilds the local text and visual indexes. Removing a
document marks it inactive and leaves its PDF on disk. Bibliographic metadata is
stored during ingestion and formatted as an IEEE-style reference by the API.

The Streamlit file imports only `httpx`, Streamlit, and the Python standard
library. Retrieval and agent code stay behind FastAPI. The UI contains only:

- active-document management, upload staging, and explicit processing;
- question and execution-mode selection;
- streamed answer text and the actual execution label;
- IEEE citations, evidence passages, figure/page crops, and a compact timeline;
- the latest mode-separated evaluation table.

## Verification

```bash
make verify
```

The current repository passes Ruff, strict mypy, 106 tests, the 85% core
coverage gate, and strict validation of all 100 benchmark examples.
