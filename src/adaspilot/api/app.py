"""FastAPI boundary for the local ADASPilot application."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from adaspilot.api.corpus import (
    corpus_needs_rebuild,
    corpus_status,
    deactivate_document,
    ingest_staged_files,
    stage_pdf,
)
from adaspilot.api.models import (
    CitationResponse,
    CorpusStatusResponse,
    DocumentActionResponse,
    IngestionJobResponse,
    IngestStagedResponse,
    QueryRequest,
    UploadResponse,
)
from adaspilot.api.runtime import RuntimeManager
from adaspilot.config import Settings, get_settings
from adaspilot.domain.citations import doi_url, format_ieee
from adaspilot.domain.ingestion import IngestionConfig
from adaspilot.domain.serving import ExecutionRecord
from adaspilot.domain.system_evaluation import SystemEvaluationReport
from adaspilot.ingestion import IngestionPipeline
from adaspilot.ingestion.io import load_documents, read_json, resolve_current_build
from adaspilot.logging import configure_logging
from adaspilot.retrieval.indexing import build_text_index
from adaspilot.retrieval.runtime import make_visual_embedder
from adaspilot.retrieval.visual import build_visual_index


class HealthResponse(BaseModel):
    """Small response returned by the health endpoint."""

    application: str
    version: str
    status: Literal["ok"]


def create_app(
    *,
    settings: Settings | None = None,
    runtime_manager: RuntimeManager | None = None,
) -> FastAPI:
    """Create the small API used by the project."""

    runtime_settings = settings or get_settings()
    configure_logging(runtime_settings)

    manager = runtime_manager or RuntimeManager(runtime_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.close()

    application = FastAPI(
        title=runtime_settings.app_name,
        version=runtime_settings.app_version,
        description="Auditable ADAS research intelligence API",
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.runtime_manager = manager
    application.state.corpus_lock = asyncio.Lock()
    application.state.ingestion_job = IngestionJobResponse()
    application.state.ingestion_task = None

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health(request: Request) -> HealthResponse:
        settings: Settings = request.app.state.settings
        return HealthResponse(
            application=settings.app_name,
            version=settings.app_version,
            status="ok",
        )

    @application.get("/corpus/status", response_model=CorpusStatusResponse, tags=["corpus"])
    async def get_corpus_status(request: Request) -> CorpusStatusResponse:
        try:
            return corpus_status(request.app.state.settings)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=f"Corpus is not ready: {exc}") from exc

    @application.post("/corpus/upload", response_model=UploadResponse, tags=["corpus"])
    async def upload_corpus_pdf(
        request: Request,
        file: Annotated[UploadFile, File(description="PDF to stage for corpus review")],
    ) -> UploadResponse:
        _reject_while_ingesting(request)
        content = await file.read(request.app.state.settings.max_upload_bytes + 1)
        try:
            return stage_pdf(
                request.app.state.settings,
                file.filename or "document.pdf",
                content,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await file.close()

    @application.delete(
        "/corpus/documents/{document_id}",
        response_model=DocumentActionResponse,
        tags=["corpus"],
    )
    async def delete_corpus_document(request: Request, document_id: str) -> DocumentActionResponse:
        _reject_while_ingesting(request)
        async with request.app.state.corpus_lock:
            try:
                result = await asyncio.to_thread(
                    deactivate_document,
                    request.app.state.settings,
                    document_id,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Document not found") from exc
            request.app.state.runtime_manager.close()
            return result

    @application.post(
        "/corpus/ingest-staged",
        response_model=IngestionJobResponse,
        status_code=202,
        tags=["corpus"],
    )
    async def process_staged_corpus(request: Request) -> IngestionJobResponse:
        current: IngestionJobResponse = request.app.state.ingestion_job
        if current.status == "running":
            return current
        request.app.state.ingestion_job = IngestionJobResponse(
            status="running",
            message="Validating staged PDFs...",
        )
        task = asyncio.create_task(_run_ingestion_job(request.app))
        request.app.state.ingestion_task = task
        started: IngestionJobResponse = request.app.state.ingestion_job
        return started

    @application.get(
        "/corpus/ingest-status",
        response_model=IngestionJobResponse,
        tags=["corpus"],
    )
    async def ingestion_status(request: Request) -> IngestionJobResponse:
        current: IngestionJobResponse = request.app.state.ingestion_job
        return current

    @application.get(
        "/corpus/documents/{document_id}/citation",
        response_model=CitationResponse,
        tags=["corpus"],
    )
    async def document_citation(
        request: Request,
        document_id: str,
        ref: int = 1,
    ) -> CitationResponse:
        if ref < 1:
            raise HTTPException(status_code=400, detail="Reference number must be positive")
        try:
            build_dir, _ = resolve_current_build(request.app.state.settings.processed_data_dir)
            document = next(
                (item for item in load_documents(build_dir) if item.document_id == document_id),
                None,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=f"Corpus is not ready: {exc}") from exc
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return CitationResponse(
            citation=format_ieee(document, ref),
            doi_url=doi_url(document),
        )

    @application.post("/query", response_model=ExecutionRecord, tags=["query"])
    async def query(request: Request, payload: QueryRequest) -> ExecutionRecord:
        _reject_while_ingesting(request)
        runner = request.app.state.runtime_manager.runner()
        return await asyncio.to_thread(runner.run, payload.question, payload.mode)

    @application.post("/query/stream", tags=["query"])
    async def stream_query(request: Request, payload: QueryRequest) -> StreamingResponse:
        _reject_while_ingesting(request)

        async def events() -> AsyncIterator[str]:
            yield _sse("accepted", {"requested_mode": payload.mode.value})
            runner = request.app.state.runtime_manager.runner()
            record = await asyncio.to_thread(runner.run, payload.question, payload.mode)
            yield _sse(
                "execution",
                {"mode": record.mode.value, "latency_ms": record.latency_ms},
            )
            for item in record.timeline:
                yield _sse("timeline", item.model_dump(mode="json"))
            chunk_size = request.app.state.settings.stream_chunk_chars
            for offset in range(0, len(record.answer.answer), chunk_size):
                yield _sse("answer", {"text": record.answer.answer[offset : offset + chunk_size]})
                await asyncio.sleep(0)
            yield _sse(
                "evidence",
                {
                    "citations": [item.model_dump(mode="json") for item in record.answer.citations],
                    "evidence": [item.model_dump(mode="json") for item in record.evidence],
                },
            )
            yield _sse("done", {"terminal_reason": record.terminal_reason})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.get(
        "/evaluation/latest",
        response_model=SystemEvaluationReport,
        tags=["evaluation"],
    )
    async def latest_evaluation(request: Request) -> SystemEvaluationReport:
        path: Path = request.app.state.settings.evaluation_report_path
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="No system report yet. Run make evaluate-system first.",
            )
        return read_json(path, SystemEvaluationReport)

    @application.get("/assets/{asset_path:path}", response_class=FileResponse, tags=["corpus"])
    async def get_asset(request: Request, asset_path: str) -> FileResponse:
        build_dir, _ = resolve_current_build(request.app.state.settings.processed_data_dir)
        root = build_dir.resolve()
        candidate = (root / asset_path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(candidate)

    return application


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _reject_while_ingesting(request: Request) -> None:
    job: IngestionJobResponse = request.app.state.ingestion_job
    if job.status == "running":
        raise HTTPException(
            status_code=409,
            detail="Corpus processing is in progress. Try again after it finishes.",
        )


async def _run_ingestion_job(application: FastAPI) -> None:
    settings: Settings = application.state.settings
    pointer_path = settings.processed_data_dir / "current.json"
    previous_pointer = pointer_path.read_text(encoding="utf-8") if pointer_path.is_file() else None

    def update_message(message: str) -> None:
        current: IngestionJobResponse = application.state.ingestion_job
        application.state.ingestion_job = current.model_copy(update={"message": message})

    async with application.state.corpus_lock:
        try:
            result: IngestStagedResponse = await asyncio.to_thread(
                ingest_staged_files,
                settings,
            )
            requires_rebuild = bool(result.ingested or result.reactivated) or corpus_needs_rebuild(
                settings
            )
            if not requires_rebuild:
                application.state.ingestion_job = IngestionJobResponse(
                    status="succeeded",
                    message="No new corpus changes were found.",
                    ingested=result.ingested,
                    reactivated=result.reactivated,
                    errors=result.errors,
                )
                return

            application.state.ingestion_job = IngestionJobResponse(
                status="running",
                message="Preparing the updated corpus...",
                ingested=result.ingested,
                reactivated=result.reactivated,
                errors=result.errors,
            )
            application.state.runtime_manager.close()
            corpus_version = await asyncio.to_thread(
                _rebuild_corpus,
                settings,
                update_message,
            )
            application.state.ingestion_job = IngestionJobResponse(
                status="succeeded",
                message="Corpus processing completed.",
                ingested=result.ingested,
                reactivated=result.reactivated,
                errors=result.errors,
                corpus_version=corpus_version,
            )
        except Exception as exc:  # Index backends may expose provider-specific failures.
            if previous_pointer is not None:
                pointer_path.write_text(previous_pointer, encoding="utf-8")
            current = application.state.ingestion_job
            application.state.ingestion_job = current.model_copy(
                update={
                    "status": "failed",
                    "message": "Corpus processing failed. The previous index is still available.",
                    "errors": [*current.errors, str(exc)],
                }
            )


def _rebuild_corpus(
    settings: Settings,
    progress: Callable[[str], None] | None = None,
) -> str:
    report = progress or (lambda _: None)
    report("Extracting text, tables, pages and figures...")
    config = IngestionConfig(
        render_dpi=settings.render_dpi,
        parent_chunk_chars=settings.parent_chunk_chars,
        child_chunk_chars=settings.child_chunk_chars,
        child_chunk_overlap=settings.child_chunk_overlap,
        minimum_ocr_chars=settings.minimum_ocr_chars,
    )
    build = IngestionPipeline(
        manifest_path=settings.corpus_manifest_path,
        raw_corpus_dir=settings.raw_corpus_dir,
        processed_dir=settings.processed_data_dir,
        config=config,
    ).run()
    report("Building the text retrieval index...")
    store, _ = build_text_index(settings)
    store.close()
    report("Building the visual retrieval index...")
    build_dir, _ = resolve_current_build(settings.processed_data_dir)
    build_visual_index(
        build_dir=build_dir,
        output_dir=settings.index_dir / f"visual_{build.corpus_version}",
        embedder=make_visual_embedder(settings),
    )
    return build.corpus_version


app = create_app()
