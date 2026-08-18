"""Cross-record benchmark validation tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from adaspilot.domain.corpus import CorpusManifest
from adaspilot.domain.evaluation import BenchmarkTargets, EvaluationExample
from adaspilot.evaluation.io import (
    load_benchmark_targets,
    load_corpus_manifest,
    load_evaluation_examples,
)
from adaspilot.evaluation.validation import validate_benchmark


def _targets() -> BenchmarkTargets:
    return BenchmarkTargets.model_validate(
        {
            "minimum_verified_total": 1,
            "maximum_verified_total": 10,
            "recommended_verified_total": 1,
            "categories": {
                "factual": {"minimum": 1, "recommended": 1},
                "table": {"minimum": 0, "recommended": 0},
                "visual": {"minimum": 0, "recommended": 0},
                "comparison": {"minimum": 0, "recommended": 0},
                "multi_step": {"minimum": 0, "recommended": 0},
                "unanswerable": {"minimum": 0, "recommended": 0},
                "adversarial": {"minimum": 0, "recommended": 0},
            },
        }
    )


def _manifest() -> CorpusManifest:
    return CorpusManifest.model_validate(
        {
            "corpus_id": "adas_test",
            "description": "A small manifest used by unit tests.",
            "documents": [
                {
                    "document_id": "doc_one",
                    "title": "Example ADAS report",
                    "source_kind": "public_url",
                    "source_uri": "https://example.com/report.pdf",
                    "status": "verified",
                    "expected_sha256": "a" * 64,
                    "page_count": 10,
                }
            ],
        }
    )


def _verified_example(document_id: str = "doc_one") -> EvaluationExample:
    return EvaluationExample.model_validate(
        {
            "question_id": "q_verified_fact",
            "question": "Which sensor is used by the example system?",
            "question_type": "factual",
            "verification_status": "verified",
            "should_answer": True,
            "expected_response_status": "answered",
            "expected_execution_mode": "direct",
            "required_modalities": ["text"],
            "reference_answer": "The system uses a forward-facing camera.",
            "grading_notes": ["Must identify the forward-facing camera."],
            "evidence": [
                {
                    "document_id": document_id,
                    "page_number": 4,
                    "supporting_excerpt": "A forward-facing camera provides input.",
                }
            ],
            "verified_by": "reviewer",
            "verified_at": datetime.now(UTC).isoformat(),
        }
    )


def test_valid_verified_example_meets_small_target() -> None:
    report = validate_benchmark(_manifest(), [_verified_example()], _targets(), require_target=True)

    assert report.valid is True
    assert report.verified_count == 1
    assert report.verified_category_counts["factual"] == 1


def test_unknown_document_reference_is_an_error() -> None:
    report = validate_benchmark(_manifest(), [_verified_example("doc_missing")], _targets())

    assert report.valid is False
    assert any(issue.code == "unknown_document_reference" for issue in report.issues)


def test_page_reference_cannot_exceed_document_length() -> None:
    example = _verified_example()
    example.evidence[0].page_number = 11

    report = validate_benchmark(_manifest(), [example], _targets())

    assert report.valid is False
    assert any(issue.code == "page_number_out_of_range" for issue in report.issues)


def test_verified_example_requires_a_verified_document() -> None:
    manifest = _manifest()
    manifest.documents[0].status = "planned"

    report = validate_benchmark(manifest, [_verified_example()], _targets())

    assert report.valid is False
    assert any(issue.code == "unverified_document_reference" for issue in report.issues)


def test_duplicate_question_text_is_rejected() -> None:
    first = _verified_example()
    second = _verified_example()
    second.question_id = "q_verified_copy"

    report = validate_benchmark(_manifest(), [first, second], _targets())

    assert report.valid is False
    assert any(issue.code == "duplicate_question_text" for issue in report.issues)


def test_missing_target_is_warning_until_strict_mode() -> None:
    relaxed = validate_benchmark(_manifest(), [], _targets(), require_target=False)
    strict = validate_benchmark(_manifest(), [], _targets(), require_target=True)

    assert relaxed.valid is True
    assert strict.valid is False


def test_committed_benchmark_reaches_minimum_target() -> None:
    root = Path(__file__).parents[2]
    manifest = load_corpus_manifest(root / "data/corpus/manifest.json")
    examples = load_evaluation_examples(root / "data/evaluation/benchmark.jsonl")
    targets = load_benchmark_targets(root / "data/evaluation/targets.json")
    report = validate_benchmark(manifest, examples, targets, require_target=True)

    assert report.valid is True
    # Local UI additions legitimately grow the manifest; the benchmark contract is
    # about validity and verified examples, not a frozen corpus size.
    assert report.document_count == len(manifest.documents)
    assert report.example_count == 100
    assert report.verified_count == 100
    assert report.draft_count == 0
    assert report.category_counts == {
        "adversarial": 10,
        "comparison": 12,
        "factual": 30,
        "multi_step": 12,
        "table": 12,
        "unanswerable": 12,
        "visual": 12,
    }
