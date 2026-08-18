"""Composition helpers for running Modules 6-8 over the real local corpus."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from adaspilot.config import Settings
from adaspilot.domain.corpus import DocumentStatus
from adaspilot.evaluation.io import load_corpus_manifest
from adaspilot.ingestion.io import (
    load_chunks,
    load_documents,
    load_pages,
    resolve_current_build,
)
from adaspilot.observability.usage import UsageTracker
from adaspilot.orchestration.direct import DirectCorrectiveWorkflow
from adaspilot.orchestration.model_adapters import (
    OpenAICompatibleActionPolicy,
    OpenAICompatiblePlanner,
    StructuredModelClient,
)
from adaspilot.orchestration.planning import (
    PlanExecuteWorkflow,
    Researcher,
    RuleBasedPlanner,
)
from adaspilot.orchestration.react import ReactResearchAgent
from adaspilot.orchestration.tools import ProjectToolbox
from adaspilot.retrieval.indexing import load_dense_retriever
from adaspilot.retrieval.qdrant_store import QdrantChunkStore
from adaspilot.retrieval.runtime import (
    make_advanced_retriever,
    make_answerer,
    make_visual_embedder,
)
from adaspilot.retrieval.visual import load_visual_index
from adaspilot.safety.guardrails import AccessPolicy, InputGuard
from adaspilot.safety.plan_validation import PlanValidator
from adaspilot.tools.web_search import search_adas_web


@dataclass
class WorkflowBundle:
    direct: DirectCorrectiveWorkflow
    react: ReactResearchAgent
    plan: PlanExecuteWorkflow
    store: QdrantChunkStore
    usage_tracker: UsageTracker
    corpus_version: str

    def close(self) -> None:
        self.store.close()


def build_workflows(
    settings: Settings,
    *,
    allowed_document_ids: set[str] | None = None,
) -> WorkflowBundle:
    dense, store, build_dir = load_dense_retriever(settings)
    advanced = make_advanced_retriever(settings, dense, build_dir)
    usage_tracker = UsageTracker(
        settings.llm_input_cost_per_million,
        settings.llm_output_cost_per_million,
    )
    answerer = make_answerer(settings, usage_tracker)
    input_guard = InputGuard(settings.max_input_chars)
    structured_client = (
        StructuredModelClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout_seconds=settings.llm_timeout_seconds,
            usage_tracker=usage_tracker,
        )
        if settings.llm_base_url and settings.llm_model and settings.llm_api_key
        else None
    )
    _, build = resolve_current_build(settings.processed_data_dir)
    visual_dir = settings.index_dir / f"visual_{build.corpus_version}"
    visual = (
        load_visual_index(visual_dir, make_visual_embedder(settings))
        if visual_dir.exists()
        else None
    )
    manifest = load_corpus_manifest(settings.corpus_manifest_path)
    active_documents = [
        document for document in manifest.documents if document.status != DocumentStatus.INACTIVE
    ]
    active_ids = {document.document_id for document in active_documents}
    access = AccessPolicy(allowed_document_ids if allowed_document_ids is not None else active_ids)
    titles = {document.document_id: document.title for document in active_documents}
    toolbox = ProjectToolbox(
        text_retriever=advanced,
        visual_index=visual,
        build_dir=build_dir,
        pages=load_pages(build_dir),
        documents=load_documents(build_dir),
        parent_chunks=load_chunks(build_dir, "parent"),
        child_chunks=load_chunks(build_dir, "child"),
        document_titles=titles,
        web_searcher=(
            partial(search_adas_web, settings=settings)
            if settings.tavily_api_key is not None
            else None
        ),
        access_policy=access,
        timeout_seconds=settings.node_timeout_seconds,
    )
    direct = DirectCorrectiveWorkflow(
        retriever=advanced,
        answerer=answerer,
        input_guard=input_guard,
        access_policy=access,
        evidence_score_threshold=settings.evidence_score_threshold,
        top_k=settings.dense_top_k,
        node_timeout_seconds=settings.node_timeout_seconds,
        recursion_limit=settings.direct_recursion_limit,
    )
    react = ReactResearchAgent(
        toolbox=toolbox,
        answerer=answerer,
        input_guard=input_guard,
        policy=(
            OpenAICompatibleActionPolicy(
                structured_client,
                allow_web_search=settings.tavily_api_key is not None,
            )
            if structured_client
            else None
        ),
        max_tool_calls=settings.react_max_tool_calls,
        recursion_limit=settings.react_recursion_limit,
    )
    plan = PlanExecuteWorkflow(
        planner=(
            OpenAICompatiblePlanner(structured_client, settings.plan_max_tasks)
            if structured_client
            else RuleBasedPlanner()
        ),
        researcher=Researcher(toolbox),
        answerer=answerer,
        input_guard=input_guard,
        plan_validator=PlanValidator(settings.plan_max_tasks, settings.plan_max_depth),
        max_parallel_workers=settings.plan_max_parallel_workers,
        max_replans=settings.plan_max_replans,
    )
    return WorkflowBundle(
        direct=direct,
        react=react,
        plan=plan,
        store=store,
        usage_tracker=usage_tracker,
        corpus_version=build.corpus_version,
    )
