"""Page-local parent and child chunk construction."""

from __future__ import annotations

import hashlib
import re

from adaspilot.domain.ingestion import (
    ChunkKind,
    ChunkRecord,
    ContentModality,
    IngestionConfig,
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def stable_id(prefix: str, *parts: object) -> str:
    source = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(source.encode()).hexdigest()[:24]}"


def _paragraphs(text: str) -> list[str]:
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]


def _group_paragraphs(text: str, maximum_chars: int) -> list[str]:
    groups: list[str] = []
    current = ""
    for paragraph in _paragraphs(text):
        if current and len(current) + len(paragraph) + 2 > maximum_chars:
            groups.append(current)
            current = ""
        if len(paragraph) > maximum_chars:
            if current:
                groups.append(current)
                current = ""
            for start in range(0, len(paragraph), maximum_chars):
                groups.append(paragraph[start : start + maximum_chars])
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        groups.append(current)
    return groups


def _child_windows(text: str, size: int, overlap: int) -> list[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_END.split(text) if sentence.strip()]
    if not sentences:
        return []
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            windows.append(current)
            tail = current[-overlap:].lstrip() if overlap else ""
            current = f"{tail} {sentence}".strip()
        else:
            current = f"{current} {sentence}".strip()
    if current:
        windows.append(current)
    return windows


def chunk_page(
    *,
    document_id: str,
    document_version: str,
    corpus_version: str,
    page_number: int,
    text: str,
    config: IngestionConfig,
) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    parents: list[ChunkRecord] = []
    children: list[ChunkRecord] = []
    for parent_index, parent_text in enumerate(
        _group_paragraphs(text, config.parent_chunk_chars), start=1
    ):
        parent_id = stable_id("parent", document_version, page_number, parent_index, parent_text)
        parents.append(
            ChunkRecord(
                chunk_id=parent_id,
                document_id=document_id,
                document_version=document_version,
                corpus_version=corpus_version,
                page_number=page_number,
                kind=ChunkKind.PARENT,
                modality=ContentModality.TEXT,
                text=parent_text,
            )
        )
        for child_index, child_text in enumerate(
            _child_windows(
                parent_text,
                config.child_chunk_chars,
                config.child_chunk_overlap,
            ),
            start=1,
        ):
            children.append(
                ChunkRecord(
                    chunk_id=stable_id("child", parent_id, child_index, child_text),
                    parent_id=parent_id,
                    document_id=document_id,
                    document_version=document_version,
                    corpus_version=corpus_version,
                    page_number=page_number,
                    kind=ChunkKind.CHILD,
                    modality=ContentModality.TEXT,
                    text=child_text,
                )
            )
    return parents, children
