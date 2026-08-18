# Benchmark curation and verification

## Objective

Maintain 100-150 source-verified questions that measure ADASPilot instead of
merely demonstrating it. The current minimum benchmark has 100 records; the
recommended later total is 120.

| Category | Minimum | Recommended |
|---|---:|---:|
| Factual | 30 | 35 |
| Table | 12 | 15 |
| Visual | 12 | 15 |
| Comparison | 12 | 15 |
| Multi-step | 12 | 15 |
| Unanswerable | 12 | 15 |
| Adversarial | 10 | 10 |
| Total | 100 | 120 |

## Verification workflow

1. Add an approved public document to `data/corpus/manifest.json` as `planned`.
2. Download it into ignored `data/corpus/raw/` and record its SHA-256 hash and page count.
3. Review its source, license/use conditions, page count, and extraction quality.
4. Change the document to `verified` only when its hash is fixed.
5. Convert a draft question into a document-specific question.
6. Add a reference answer written from the source evidence.
7. Add every relevant document ID and one-indexed page number.
8. For visual questions, add an asset ID or bounding box.
9. Add concise grading notes that describe required answer points.
10. Set `verified_by`, `verified_at`, and `verification_status=verified`.
11. Before publishing final model metrics, have the project owner spot-check difficult, visual, and comparison examples.
12. Run `make validate-data-strict`.

The repository contains five verified source files and 100 source-reviewed
questions at the minimum target distribution. The `verified_by` field records
the repeatable review pass rather than attributing it to a person. The project
owner should spot-check the held-out and visual records before freezing final
resume metrics.

## Split policy

- Use `development` while designing and debugging retrieval.
- Freeze a held-out `test` subset before final tuning.
- Do not inspect or tune repeatedly on test failures.
- Keep near-duplicate questions within the same split.

## Evidence policy

- Page numbers are one-indexed as shown to a human reader.
- An answerable verified example requires at least one evidence reference.
- A verified visual example requires an asset ID or bounding box.
- Verified unanswerable/adversarial examples must contain no supporting evidence.
- Do not use model-generated answers as ground truth without human source review.
- Do not mark an example verified merely because a model and the reference agree.

## Quality review

Check that each question:

- Has one clear interpretation or explicitly lists acceptable alternatives
- Requires the declared modalities
- Is answerable only from the referenced evidence when `should_answer=true`
- Does not leak the answer through its wording
- Has no unsupported numerical precision
- Uses a justified direct, ReAct, or plan-and-execute expectation
- Represents a realistic ADAS research task
