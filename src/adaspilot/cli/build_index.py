"""Build the local Qdrant dense index."""

from __future__ import annotations

from adaspilot.config import get_settings
from adaspilot.retrieval.indexing import build_text_index


def main() -> None:
    settings = get_settings()
    store, chunks = build_text_index(settings)
    try:
        print(f"Indexed {len(chunks)} child chunks into {store.collection_name}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
