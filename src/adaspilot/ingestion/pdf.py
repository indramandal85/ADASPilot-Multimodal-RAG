"""Page-aware PDF extraction using PyMuPDF and Tesseract."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
import pytesseract  # type: ignore[import-untyped]
from PIL import Image

from adaspilot.domain.ingestion import (
    AssetKind,
    IngestionConfig,
    OCRToken,
    PageBox,
    PageRecord,
    TableRecord,
    VisualAssetRecord,
)
from adaspilot.ingestion.chunking import stable_id

LOGGER = logging.getLogger(__name__)
_CAPTION = re.compile(r"^(fig(?:ure)?\.?|table)\s+\d+[.:]\s+", re.IGNORECASE)


@dataclass(frozen=True)
class ExtractedPage:
    page: PageRecord
    tables: list[TableRecord]
    visual_assets: list[VisualAssetRecord]


def _box(rect: pymupdf.Rect | tuple[float, float, float, float]) -> PageBox:
    values = rect if isinstance(rect, tuple) else (rect.x0, rect.y0, rect.x1, rect.y1)
    return PageBox(x0=values[0], y0=values[1], x1=values[2], y1=values[3])


def _relative(path: Path, build_dir: Path) -> str:
    return path.relative_to(build_dir).as_posix()


def _text_blocks(page: pymupdf.Page) -> list[tuple[float, float, float, float, str]]:
    blocks: list[tuple[float, float, float, float, str]] = []
    for raw in page.get_text("blocks", sort=True):
        text = str(raw[4]).strip()
        if text:
            blocks.append((float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]), text))
    return blocks


def _table_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    lines = [
        "| " + " | ".join(cell.replace("|", "\\|") for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in normalized[1:]
    )
    return "\n".join(lines)


def _extract_tables(
    page: pymupdf.Page,
    document_id: str,
    page_number: int,
    captions: list[tuple[pymupdf.Rect, str]],
) -> list[TableRecord]:
    records: list[TableRecord] = []
    try:
        found = page.find_tables()
    except Exception as exc:  # pragma: no cover - defensive for malformed PDFs
        LOGGER.warning("table extraction failed for %s page %s: %s", document_id, page_number, exc)
        return records
    for index, table in enumerate(found.tables, start=1):
        raw_rows = table.extract()
        rows = [
            ["" if cell is None else " ".join(str(cell).split()) for cell in row]
            for row in raw_rows
        ]
        if not rows or not any(any(cell for cell in row) for row in rows):
            continue
        rect = pymupdf.Rect(table.bbox)
        caption = next(
            (text for box, text in captions if 0 <= rect.y0 - box.y1 <= 80),
            None,
        )
        table_id = stable_id("table", document_id, page_number, index, rows)
        records.append(
            TableRecord(
                table_id=table_id,
                document_id=document_id,
                page_number=page_number,
                bounding_box=_box(rect),
                caption=caption,
                rows=rows,
                markdown=_table_markdown(rows),
            )
        )
    return records


def _asset_kind(caption: str) -> AssetKind:
    lowered = caption.lower()
    if lowered.startswith("table"):
        return AssetKind.TABLE_IMAGE
    if "plot" in lowered or "curve" in lowered or "results" in lowered:
        return AssetKind.PLOT
    if "architecture" in lowered or "pipeline" in lowered or "diagram" in lowered:
        return AssetKind.DIAGRAM
    return AssetKind.FIGURE


def _ocr_image(image: Image.Image, clip: pymupdf.Rect) -> tuple[str | None, list[OCRToken]]:
    data: dict[str, list[Any]] = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )
    tokens: list[OCRToken] = []
    collected: list[str] = []
    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        confidence = float(data["conf"][index])
        if not text or confidence < 30:
            continue
        left = float(data["left"][index])
        top = float(data["top"][index])
        width = float(data["width"][index])
        height = float(data["height"][index])
        x_scale = clip.width / image.width
        y_scale = clip.height / image.height
        token_box = PageBox(
            x0=clip.x0 + left * x_scale,
            y0=clip.y0 + top * y_scale,
            x1=clip.x0 + (left + width) * x_scale,
            y1=clip.y0 + (top + height) * y_scale,
        )
        tokens.append(OCRToken(text=text, confidence=confidence, bounding_box=token_box))
        collected.append(text)
    ocr_text = " ".join(collected).strip()
    return (ocr_text or None), tokens


def _caption_visuals(
    *,
    page: pymupdf.Page,
    document_id: str,
    page_number: int,
    captions: list[tuple[pymupdf.Rect, str]],
    build_dir: Path,
    document_asset_dir: Path,
    config: IngestionConfig,
) -> list[VisualAssetRecord]:
    assets: list[VisualAssetRecord] = []
    scale = config.render_dpi / 72
    for caption_box, caption in captions:
        kind = _asset_kind(caption)
        if kind == AssetKind.TABLE_IMAGE:
            top = max(0.0, caption_box.y0 - 4)
            bottom = min(page.rect.height, caption_box.y1 + page.rect.height * 0.42)
        else:
            top = max(0.0, caption_box.y0 - page.rect.height * 0.42)
            bottom = min(page.rect.height, caption_box.y1 + 4)
        clip = pymupdf.Rect(0, top, page.rect.width, bottom)
        if clip.height < 40:
            continue
        asset_id = stable_id("asset", document_id, page_number, caption)
        image_path = document_asset_dir / f"{asset_id}.png"
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
        pixmap.save(image_path)
        ocr_text: str | None = None
        ocr_tokens: list[OCRToken] = []
        if config.enable_ocr:
            with Image.open(image_path) as image:
                ocr_text, ocr_tokens = _ocr_image(image.convert("RGB"), clip)
        assets.append(
            VisualAssetRecord(
                asset_id=asset_id,
                document_id=document_id,
                page_number=page_number,
                kind=kind,
                bounding_box=_box(clip),
                image_path=_relative(image_path, build_dir),
                caption=caption,
                ocr_text=ocr_text,
                ocr_tokens=ocr_tokens,
            )
        )
    return assets


def _embedded_visuals(
    *,
    page: pymupdf.Page,
    document_id: str,
    page_number: int,
    build_dir: Path,
    document_asset_dir: Path,
) -> list[VisualAssetRecord]:
    assets: list[VisualAssetRecord] = []
    page_dict: dict[str, Any] = page.get_text("dict")
    for index, block in enumerate(page_dict.get("blocks", []), start=1):
        if block.get("type") != 1 or "image" not in block:
            continue
        rect = pymupdf.Rect(block["bbox"])
        if rect.width < 80 or rect.height < 60 or rect.get_area() < page.rect.get_area() * 0.015:
            continue
        extension = str(block.get("ext", "png"))
        asset_id = stable_id("image", document_id, page_number, index, rect)
        image_path = document_asset_dir / f"{asset_id}.{extension}"
        image_path.write_bytes(bytes(block["image"]))
        assets.append(
            VisualAssetRecord(
                asset_id=asset_id,
                document_id=document_id,
                page_number=page_number,
                kind=AssetKind.EMBEDDED_IMAGE,
                bounding_box=_box(rect),
                image_path=_relative(image_path, build_dir),
            )
        )
    return assets


def extract_page(
    *,
    page: pymupdf.Page,
    document_id: str,
    document_version: str,
    corpus_version: str,
    build_dir: Path,
    config: IngestionConfig,
) -> ExtractedPage:
    page_number = page.number + 1
    page_dir = build_dir / "pages" / document_id
    asset_dir = build_dir / "assets" / document_id
    page_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    blocks = _text_blocks(page)
    text = "\n\n".join(block[4] for block in blocks)
    ocr_used = False
    scale = config.render_dpi / 72
    page_image_path = page_dir / f"page_{page_number:04d}.png"
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    pixmap.save(page_image_path)
    if config.enable_ocr and len(text) < config.minimum_ocr_chars:
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
            ocr_text = pytesseract.image_to_string(image, config="--psm 6").strip()
        if ocr_text:
            text = ocr_text
            ocr_used = True

    captions: list[tuple[pymupdf.Rect, str]] = []
    for x0, y0, x1, y1, block_text in blocks:
        lines = [line.strip() for line in block_text.splitlines() if line.strip()]
        if lines and len(block_text) <= 700 and _CAPTION.match(lines[0]):
            captions.append((pymupdf.Rect(x0, y0, x1, y1), " ".join(block_text.split())))

    tables = (
        _extract_tables(page, document_id, page_number, captions) if config.extract_tables else []
    )
    assets: list[VisualAssetRecord] = []
    if config.extract_visuals:
        assets.extend(
            _caption_visuals(
                page=page,
                document_id=document_id,
                page_number=page_number,
                captions=captions,
                build_dir=build_dir,
                document_asset_dir=asset_dir,
                config=config,
            )
        )
        assets.extend(
            _embedded_visuals(
                page=page,
                document_id=document_id,
                page_number=page_number,
                build_dir=build_dir,
                document_asset_dir=asset_dir,
            )
        )

    page_record = PageRecord(
        page_id=stable_id("page", document_version, page_number),
        document_id=document_id,
        document_version=document_version,
        corpus_version=corpus_version,
        page_number=page_number,
        width=page.rect.width,
        height=page.rect.height,
        text=text,
        text_block_count=len(blocks),
        rendered_image_path=_relative(page_image_path, build_dir),
        ocr_used=ocr_used,
        table_ids=[table.table_id for table in tables],
        visual_asset_ids=[asset.asset_id for asset in assets],
    )
    return ExtractedPage(page=page_record, tables=tables, visual_assets=assets)
