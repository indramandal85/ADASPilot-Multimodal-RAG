# Evaluation data

Files:

- `benchmark.jsonl` - one versioned `EvaluationExample` per line
- `targets.json` - required category distribution and 100-150 verified target

The file contains 100 source-verified examples: 30 factual, 12 table, 12
visual, 12 comparison, 12 multi-step, 12 unanswerable, and 10 adversarial.
Every answerable record has page evidence; visual records have an asset ID.
The fixed split contains 80 development and 20 test examples.

Run `make validate-data` during development. Run `make validate-data-strict`
before changing benchmark or corpus files.
