# Modules 2-5 implementation guide

This document is the interview-level walkthrough of the implemented pipeline.

## Module 2: deterministic multimodal ingestion

`IngestionPipeline` reads the verified corpus manifest, checks each PDF hash and
page count, and deduplicates identical files. Document versions depend on file
content; the corpus version depends on document versions plus extraction
settings. Running the same inputs twice therefore resolves to the same build.

For each PDF page, PyMuPDF extracts page-space text blocks and tables, renders a
stable PNG, finds strict caption blocks, crops likely figure/table regions, and
extracts embedded images. Tesseract is used only when page text is too sparse.
OCR tokens retain PDF-space bounding boxes.

Text is stored twice:

- Parent chunks preserve a readable paragraph-sized context.
- Overlapping child chunks give the retriever more precise search units.

Table Markdown and visual captions/OCR are also searchable chunks. Every chunk
retains document ID, version, corpus version, page number, modality, parent ID,
and optional table/asset ID.

Generated build files:

```text
data/processed/<corpus_version>/
├── corpus.json
├── documents.jsonl
├── pages.jsonl
├── tables.jsonl
├── visual_assets.jsonl
├── parent_chunks.jsonl
├── child_chunks.jsonl
├── pages/*.png
└── assets/*.{png,jpg}
```

## Module 3: dense baseline

The baseline embeds child chunks with BGE and writes them to embedded local
Qdrant. Retrieval is cosine top-k. The no-LLM answer baseline selects relevant
sentences and returns a typed answer with page citations. An optional single
OpenAI-compatible adapter supports local or hosted text/vision generation
without tying the core project to one provider.

The baseline is intentionally limited: no agents, query rewriting, HyDE,
reranking, or corrective loop. Its job is to expose the starting failure modes.

## Module 4: controlled text retrieval

All configurations use the same development questions:

1. E0: BGE dense retrieval.
2. E1: dense plus BM25, fused with reciprocal-rank fusion.
3. E2: E1 candidates reranked with a cross-encoder.
4. E3: selected child hits expanded to their parent context.
5. E4: E3 plus conditional HyDE when retrieval confidence is low.

E1-E3 improved Recall@5 and remain in the selected pipeline. E4 reduced it by
0.063, so `ADASPILOT_ENABLE_HYDE=false` is the default. The code is retained to
retest HyDE on specific failure categories rather than claiming it always
helps.

## Module 5: visual retrieval

The visual index contains both full pages and extracted visual assets. The
default CLIP backend makes local development practical. The optional ColPali
adapter creates multi-vector page/query embeddings and scores them with MaxSim.

The components map directly to three modes:

- `VisualIndex.search`: visual-only retrieval.
- The normal text retriever: text-only retrieval, including captions/OCR chunks.
- `FusedMultimodalRetriever`: visual-query detection followed by text/visual RRF.

Nonvisual questions stay on the text path. Visual questions search both paths.
Only the retrieved image paths are sent to the optional vision generator, and
the structured answer cites the exact document page and asset when available.

The current visual metrics measure relevant-page Recall@1/5 and a deterministic
grounding proxy: whether a retrieved visual page supports the citation target.
The CLIP score is a baseline for later caption/OCR and ColPali ablations.

## What to explain in an interview

1. Why the benchmark was built before optimizing retrieval.
2. Why child chunks are searched but parent chunks are returned.
3. Why RRF is safer than mixing dense, BM25, and visual raw scores.
4. Why reranking only a small candidate pool controls latency.
5. Why HyDE was disabled even though it is a popular advanced-RAG technique.
6. Why pages are indexed even when figure segmentation is imperfect.
7. Why visual claims receive page/asset citations and the vision model sees only
   retrieved evidence.

