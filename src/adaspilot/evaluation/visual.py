"""Visual retrieval and citation-grounding metrics."""

from __future__ import annotations

from collections.abc import Sequence

from adaspilot.domain.evaluation import EvaluationExample, QuestionType
from adaspilot.domain.retrieval import VisualMetrics
from adaspilot.retrieval.visual import VisualIndex


def evaluate_visual_retrieval(
    index: VisualIndex,
    examples: Sequence[EvaluationExample],
) -> VisualMetrics:
    visual_examples = [
        example
        for example in examples
        if example.question_type == QuestionType.VISUAL and example.should_answer
    ]
    ranks: list[int | None] = []
    grounded = 0
    for example in visual_examples:
        relevant = {(item.document_id, item.page_number) for item in example.evidence}
        hits = index.search(example.question, 5)
        rank = next(
            (
                position
                for position, hit in enumerate(hits, start=1)
                if (hit.document_id, hit.page_number) in relevant
            ),
            None,
        )
        ranks.append(rank)
        grounded += int(rank is not None)
    total = len(ranks)
    if not total:
        return VisualMetrics(
            query_count=0,
            visual_recall_at_1=0,
            visual_recall_at_5=0,
            multimodal_faithfulness=0,
        )
    return VisualMetrics(
        query_count=total,
        visual_recall_at_1=sum(rank == 1 for rank in ranks) / total,
        visual_recall_at_5=sum(rank is not None and rank <= 5 for rank in ranks) / total,
        multimodal_faithfulness=grounded / total,
    )
