"""Build the page-and-figure visual index."""

from __future__ import annotations

from adaspilot.config import get_settings
from adaspilot.ingestion.io import resolve_current_build
from adaspilot.retrieval.runtime import make_visual_embedder
from adaspilot.retrieval.visual import build_visual_index


def main() -> None:
    settings = get_settings()
    build_dir, build = resolve_current_build(settings.processed_data_dir)
    output_dir = settings.index_dir / f"visual_{build.corpus_version}"
    index = build_visual_index(
        build_dir=build_dir,
        output_dir=output_dir,
        embedder=make_visual_embedder(settings),
    )
    print(f"Indexed {len(index.records)} page and figure images into {output_dir}")


if __name__ == "__main__":
    main()
