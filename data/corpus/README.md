# Corpus manifest

`manifest.json` is the authoritative registry of documents allowed in the
benchmark. Module 1 fixes five public sources by URL, SHA-256 hash, and page
count. Local PDF copies are kept under ignored `data/corpus/raw/`.

Each document progresses through:

```text
planned -> downloaded -> verified
```

A document becomes `verified` only after its SHA-256 hash and page count are
recorded. Later ingestion modules will reuse these values for deduplication and
corpus versioning.

Do not add private employer documents, copyrighted material without an
appropriate use basis, or files containing personal/sensitive information.
