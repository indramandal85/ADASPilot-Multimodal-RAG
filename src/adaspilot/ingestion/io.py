"""Readers and writers for generated corpus manifests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from adaspilot.domain.ingestion import (
    ChunkRecord,
    CorpusBuild,
    DocumentRecord,
    PageRecord,
    TableRecord,
    VisualAssetRecord,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")


def read_json(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, model_type: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(model_type.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid record in {path}:{line_number}: {exc}") from exc
    return records


def resolve_current_build(processed_dir: Path) -> tuple[Path, CorpusBuild]:
    pointer_path = processed_dir / "current.json"
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    version = str(payload["corpus_version"])
    build_dir = processed_dir / version
    return build_dir, read_json(build_dir / "corpus.json", CorpusBuild)


def load_documents(build_dir: Path) -> list[DocumentRecord]:
    return read_jsonl(build_dir / "documents.jsonl", DocumentRecord)


def load_pages(build_dir: Path) -> list[PageRecord]:
    return read_jsonl(build_dir / "pages.jsonl", PageRecord)


def load_tables(build_dir: Path) -> list[TableRecord]:
    return read_jsonl(build_dir / "tables.jsonl", TableRecord)


def load_visual_assets(build_dir: Path) -> list[VisualAssetRecord]:
    return read_jsonl(build_dir / "visual_assets.jsonl", VisualAssetRecord)


def load_chunks(build_dir: Path, kind: str = "child") -> list[ChunkRecord]:
    filename = "parent_chunks.jsonl" if kind == "parent" else "child_chunks.jsonl"
    return read_jsonl(build_dir / filename, ChunkRecord)
