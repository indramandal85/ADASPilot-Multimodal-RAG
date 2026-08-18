"""Optional RAGAS judge metrics kept separate from deterministic evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from statistics import fmean

from adaspilot.config import Settings
from adaspilot.domain.evaluation import EvaluationExample, Modality
from adaspilot.domain.serving import ExecutionRecord
from adaspilot.domain.system_evaluation import (
    SemanticEvaluationStatus,
    SemanticMetrics,
)


def _average(values: list[float]) -> float | None:
    return fmean(values) if values else None


class RagasEvaluator:
    """Runs current collections-based RAGAS metrics only when a judge is configured."""

    def __init__(self, settings: Settings, build_dir: Path) -> None:
        self.settings = settings
        self.build_dir = build_dir

    def evaluate(
        self,
        pairs: list[tuple[EvaluationExample, ExecutionRecord]],
    ) -> SemanticMetrics:
        if not (
            self.settings.ragas_judge_model
            and self.settings.llm_base_url
            and self.settings.llm_api_key
        ):
            return SemanticMetrics(
                status=SemanticEvaluationStatus.NOT_CONFIGURED,
                detail=(
                    "Set ADASPILOT_RAGAS_JUDGE_MODEL and the ADASPILOT_LLM_* settings "
                    "to run model-judged RAGAS metrics."
                ),
            )
        if importlib.util.find_spec("ragas") is None or importlib.util.find_spec("openai") is None:
            return SemanticMetrics(
                status=SemanticEvaluationStatus.PACKAGE_MISSING,
                detail="Install the optional evaluation dependencies with make install-evaluation.",
            )
        try:
            return self._run(pairs)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return SemanticMetrics(
                status=SemanticEvaluationStatus.FAILED,
                detail=f"RAGAS evaluation failed safely: {exc}",
            )

    def _run(
        self,
        pairs: list[tuple[EvaluationExample, ExecutionRecord]],
    ) -> SemanticMetrics:
        from openai import AsyncOpenAI  # type: ignore[import-not-found]
        from ragas.llms import llm_factory  # type: ignore[import-not-found]
        from ragas.metrics.collections import (  # type: ignore[import-not-found]
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            MultiModalFaithfulness,
        )

        assert self.settings.llm_api_key is not None
        client = AsyncOpenAI(
            api_key=self.settings.llm_api_key.get_secret_value(),
            base_url=self.settings.llm_base_url,
        )
        llm = llm_factory(self.settings.ragas_judge_model, client=client)
        context_precision = ContextPrecision(llm=llm)
        context_recall = ContextRecall(llm=llm)
        faithfulness = Faithfulness(llm=llm)
        multimodal = MultiModalFaithfulness(llm=llm)
        precision_scores: list[float] = []
        recall_scores: list[float] = []
        faithfulness_scores: list[float] = []
        multimodal_scores: list[float] = []
        for example, record in pairs:
            if not example.should_answer or not example.reference_answer:
                continue
            contexts = [item.text for item in record.evidence[:8] if item.text]
            if not contexts:
                continue
            precision_scores.append(
                float(
                    context_precision.score(
                        user_input=example.question,
                        reference=example.reference_answer,
                        retrieved_contexts=contexts,
                    ).value
                )
            )
            recall_scores.append(
                float(
                    context_recall.score(
                        user_input=example.question,
                        reference=example.reference_answer,
                        retrieved_contexts=contexts,
                    ).value
                )
            )
            faithfulness_scores.append(
                float(
                    faithfulness.score(
                        user_input=example.question,
                        response=record.answer.answer,
                        retrieved_contexts=contexts,
                    ).value
                )
            )
            if Modality.VISUAL in example.required_modalities:
                visual_contexts = list(contexts)
                visual_contexts.extend(
                    str((self.build_dir / item.image_path).resolve())
                    for item in record.evidence[:8]
                    if item.image_path and (self.build_dir / item.image_path).is_file()
                )
                multimodal_scores.append(
                    float(
                        multimodal.score(
                            response=record.answer.answer,
                            retrieved_contexts=visual_contexts,
                        ).value
                    )
                )
        return SemanticMetrics(
            status=SemanticEvaluationStatus.COMPLETED,
            context_precision=_average(precision_scores),
            context_recall=_average(recall_scores),
            faithfulness=_average(faithfulness_scores),
            multimodal_faithfulness=_average(multimodal_scores),
            detail=f"RAGAS collections API evaluated {len(faithfulness_scores)} answers.",
        )
