"""Structured answers and benchmark metric tests."""

from __future__ import annotations

from adaspilot.domain.evaluation import EvaluationExample
from adaspilot.domain.retrieval import SearchHit
from adaspilot.evaluation.retrieval import evaluate_answers, evaluate_retriever, token_f1
from adaspilot.retrieval.answers import ExtractiveAnswerer


class FixedRetriever:
    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        if "missing" in query:
            return []
        return [
            SearchHit(
                record_id="child_123",
                score=0.9,
                document_id="sample_doc",
                page_number=2,
                text="Radar measures range and velocity for vehicle perception.",
                source="fixed",
            )
        ][:top_k]


def _examples() -> list[EvaluationExample]:
    return [
        EvaluationExample.model_validate(
            {
                "question_id": "q_answerable",
                "question": "What does radar measure?",
                "question_type": "factual",
                "verification_status": "verified",
                "should_answer": True,
                "expected_response_status": "answered",
                "expected_execution_mode": "direct",
                "required_modalities": ["text"],
                "reference_answer": "Radar measures range and velocity.",
                "grading_notes": ["Require both measurements."],
                "evidence": [
                    {
                        "document_id": "sample_doc",
                        "page_number": 2,
                        "supporting_excerpt": "Radar measures range and velocity.",
                    }
                ],
                "verified_by": "test",
                "verified_at": "2026-08-17T00:00:00Z",
            }
        ),
        EvaluationExample.model_validate(
            {
                "question_id": "q_missing",
                "question": "Which missing private value is requested?",
                "question_type": "unanswerable",
                "verification_status": "verified",
                "should_answer": False,
                "expected_response_status": "abstained",
                "expected_execution_mode": "direct",
                "grading_notes": ["Must abstain."],
                "verified_by": "test",
                "verified_at": "2026-08-17T00:00:00Z",
            }
        ),
    ]


def test_structured_answer_and_metrics() -> None:
    retriever = FixedRetriever()
    answerer = ExtractiveAnswerer()
    answer = answerer.answer("What does radar measure?", retriever.retrieve("radar", 3))

    assert answer.citations[0].page_number == 2
    assert token_f1("radar range", "radar range") == 1
    retrieval = evaluate_retriever(retriever, _examples())
    answers = evaluate_answers(retriever, answerer, _examples())
    assert retrieval.recall_at_1 == 1
    assert retrieval.mean_reciprocal_rank == 1
    assert answers.citation_recall == 1
    assert answers.abstention_accuracy == 1
