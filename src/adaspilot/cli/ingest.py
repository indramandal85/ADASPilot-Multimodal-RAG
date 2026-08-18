"""CLI for deterministic multimodal ingestion."""

from __future__ import annotations

import argparse

from adaspilot.config import get_settings
from adaspilot.domain.ingestion import IngestionConfig
from adaspilot.ingestion import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the verified ADAS PDF corpus")
    parser.add_argument("--no-ocr", action="store_true", help="skip Tesseract OCR")
    arguments = parser.parse_args()
    settings = get_settings()
    config = IngestionConfig(
        render_dpi=settings.render_dpi,
        parent_chunk_chars=settings.parent_chunk_chars,
        child_chunk_chars=settings.child_chunk_chars,
        child_chunk_overlap=settings.child_chunk_overlap,
        minimum_ocr_chars=settings.minimum_ocr_chars,
        enable_ocr=not arguments.no_ocr,
    )
    build = IngestionPipeline(
        manifest_path=settings.corpus_manifest_path,
        raw_corpus_dir=settings.raw_corpus_dir,
        processed_dir=settings.processed_data_dir,
        config=config,
    ).run()
    print(build.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
