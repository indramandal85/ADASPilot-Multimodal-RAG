"""Evidence-board-only plan synthesis tests."""

from __future__ import annotations

from adaspilot.domain.orchestration import EvidenceItem
from adaspilot.orchestration.planning import select_synthesis_evidence


def _evidence(identifier: str, task_id: str, page: int, score: float) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=identifier,
        document_id=f"doc_{task_id}",
        page_number=page,
        text=f"Verified evidence for {task_id}.",
        source="test",
        tool_name="hybrid_text_search",
        task_id=task_id,
        record_id=identifier,
        support_score=score,
    )


def test_synthesis_selection_represents_each_task_with_exact_board_records() -> None:
    board = [
        _evidence("left_weak", "task_left", 3, 0.2),
        _evidence("left_best", "task_left", 7, 0.9),
        _evidence("right_best", "task_right", 12, 0.8),
        _evidence("right_extra", "task_right", 14, 0.7),
    ]

    selected = select_synthesis_evidence(board, ["task_left", "task_right"])

    assert [item.evidence_id for item in selected[:2]] == ["left_best", "right_best"]
    board_keys = {(item.document_id, item.page_number, item.record_id) for item in board}
    assert all(
        (item.document_id, item.page_number, item.record_id) in board_keys for item in selected
    )
