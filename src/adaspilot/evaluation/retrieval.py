"""Retrieval and structured-answer metrics for the verified benchmark."""

from __future__ import annotations

import re
from collections.abc import Sequence

from adaspilot.domain.evaluation import EvaluationExample
from adaspilot.domain.retrieval import AnswerMetrics, AnswerStatus, RetrievalMetrics
from adaspilot.retrieval.answers import Answerer
from adaspilot.retrieval.text import Retriever


def _relevant_pages(example: EvaluationExample) -> set[tuple[str, int]]:
    return {(item.document_id, item.page_number) for item in example.evidence}


def _first_relevant_rank(
    relevant: set[tuple[str, int]],
    ranked: Sequence[tuple[str, int]],
) -> int | None:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return rank
    return None


def evaluate_retriever(
    retriever: Retriever,
    examples: Sequence[EvaluationExample],
    *,
    top_k: int = 10,
) -> RetrievalMetrics:
    answerable = [example for example in examples if example.should_answer and example.evidence]
    ranks: list[int | None] = []
    for example in answerable:
        hits = retriever.retrieve(example.question, top_k)
        ranked = [(hit.document_id, hit.page_number) for hit in hits]
        ranks.append(_first_relevant_rank(_relevant_pages(example), ranked))
    total = len(ranks)
    if total == 0:
        return RetrievalMetrics(
            query_count=0,
            recall_at_1=0,
            recall_at_5=0,
            recall_at_10=0,
            mean_reciprocal_rank=0,
        )
    return RetrievalMetrics(
        query_count=total,
        recall_at_1=sum(rank is not None and rank <= 1 for rank in ranks) / total,
        recall_at_5=sum(rank is not None and rank <= 5 for rank in ranks) / total,
        recall_at_10=sum(rank is not None and rank <= 10 for rank in ranks) / total,
        mean_reciprocal_rank=sum(0 if rank is None else 1 / rank for rank in ranks) / total,
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(predicted: str, reference: str) -> float:
    predicted_tokens = _tokens(predicted)
    reference_tokens = _tokens(reference)
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    predicted_counts = {token: predicted_tokens.count(token) for token in set(predicted_tokens)}
    reference_counts = {token: reference_tokens.count(token) for token in set(reference_tokens)}
    overlap = sum(
        min(count, reference_counts.get(token, 0)) for token, count in predicted_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answers(
    retriever: Retriever,
    answerer: Answerer,
    examples: Sequence[EvaluationExample],
    *,
    top_k: int = 8,
) -> AnswerMetrics:
    f1_scores: list[float] = []
    citation_scores: list[float] = []
    abstention_scores: list[float] = []
    for example in examples:
        hits = retriever.retrieve(example.question, top_k)
        answer = answerer.answer(example.question, hits)
        if example.reference_answer:
            f1_scores.append(token_f1(answer.answer, example.reference_answer))
        if example.should_answer:
            relevant = _relevant_pages(example)
            cited = {(item.document_id, item.page_number) for item in answer.citations}
            citation_scores.append(float(bool(relevant & cited)))
        else:
            abstention_scores.append(float(answer.status == AnswerStatus.ABSTAINED))
    return AnswerMetrics(
        answer_count=len(examples),
        token_f1=sum(f1_scores) / max(1, len(f1_scores)),
        citation_recall=sum(citation_scores) / max(1, len(citation_scores)),
        abstention_accuracy=sum(abstention_scores) / max(1, len(abstention_scores)),
    )
