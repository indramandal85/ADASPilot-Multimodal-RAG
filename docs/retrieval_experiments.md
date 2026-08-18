# Retrieval experiment report

## Frozen setup

- Corpus version: `corpus_1d9f4eedfa93d418`
- Search units: 1,312 child chunks with parent links
- Embedding model: `BAAI/bge-small-en-v1.5`
- Reranker: `Xenova/ms-marco-MiniLM-L-6-v2`
- Evaluation: 63 answerable development questions with page evidence
- Primary promotion metric: Recall@5
- Minimum promotion gain: 0.01 absolute Recall@5

## Results and decision

| ID | Change | Recall@5 | Delta | Retained |
|---|---|---:|---:|---|
| E0 | Dense baseline | 0.698 | - | yes |
| E1 | Add BM25 and RRF | 0.794 | +0.095 | yes |
| E2 | Add cross-encoder reranking | 0.857 | +0.063 | yes |
| E3 | Add parent context | 0.873 | +0.016 | yes |
| E4 | Add conditional HyDE | 0.810 | -0.063 | no |

E3 is the selected pipeline. E4 is not promoted because it lowered Recall@1,
Recall@5, Recall@10, and MRR. This experiment is more credible than listing
HyDE as a feature without measuring it.

Raw outputs are saved in `data/results/text_experiments.json`. Regenerate them
with `make experiments` after rebuilding the corpus and dense index.

## Baseline answer limitations

The deterministic extractive baseline scored token F1 0.154, citation recall
0.603, and abstention accuracy 0.0. It retrieves for every question and cannot
yet grade weak evidence. These limitations define the work for later generation,
corrective retrieval, guardrail, and agent modules; they are not hidden by an
LLM judge at this stage.

