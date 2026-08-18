"""Stable domain contracts shared across ADASPilot modules."""

from adaspilot.domain.corpus import CorpusDocument, CorpusManifest
from adaspilot.domain.evaluation import (
    BenchmarkTargets,
    EvaluationExample,
    QuestionType,
    VerificationStatus,
)
from adaspilot.domain.ingestion import ChunkRecord, CorpusBuild, PageRecord, VisualAssetRecord
from adaspilot.domain.orchestration import DirectResult, PlanResult, ReactResult
from adaspilot.domain.retrieval import SearchHit, StructuredAnswer

__all__ = [
    "BenchmarkTargets",
    "ChunkRecord",
    "CorpusDocument",
    "CorpusBuild",
    "CorpusManifest",
    "DirectResult",
    "EvaluationExample",
    "PageRecord",
    "PlanResult",
    "QuestionType",
    "ReactResult",
    "SearchHit",
    "StructuredAnswer",
    "VerificationStatus",
    "VisualAssetRecord",
]
