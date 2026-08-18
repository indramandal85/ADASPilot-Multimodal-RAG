"""Small IEEE-style formatter for document evidence."""

from __future__ import annotations

import re
from pathlib import Path

from adaspilot.domain.ingestion import DocumentRecord


def _shorten_name(full_name: str) -> str:
    parts = full_name.strip().split()
    if not parts:
        return full_name
    if len(parts) == 1 or any(character.isdigit() for character in full_name):
        return full_name
    # Preserve organization names instead of turning "Euro NCAP" into "E. NCAP".
    if any(part.isupper() and len(part) > 1 for part in parts):
        return full_name
    initials = " ".join(f"{part[0].upper()}." for part in parts[:-1] if part)
    return f"{initials} {parts[-1]}".strip()


def _format_authors(authors: list[str]) -> str | None:
    if not authors:
        return None
    shortened = [_shorten_name(author) for author in authors]
    if len(shortened) > 3:
        return f"{shortened[0]} et al."
    if len(shortened) == 1:
        return shortened[0]
    return f"{', '.join(shortened[:-1])} and {shortened[-1]}"


def format_ieee(document: DocumentRecord, reference_number: int) -> str:
    """Return a concise IEEE-style reference with safe metadata fallbacks."""

    if reference_number < 1:
        raise ValueError("reference_number must be positive")
    metadata = document.citation_metadata
    if metadata is None or not any(
        [metadata.title, metadata.authors, metadata.year, metadata.venue, metadata.doi]
    ):
        title = Path(document.source_path).stem.replace("_", " ").strip().title()
        return f"[{reference_number}] {title}."

    parts: list[str] = []
    author_text = _format_authors(metadata.authors)
    if author_text:
        parts.append(author_text)
    title = (metadata.title or Path(document.source_path).stem.replace("_", " ")).strip()
    parts.append(f'"{title}"')
    if metadata.venue:
        parts.append(metadata.venue.strip())
    if metadata.year:
        parts.append(str(metadata.year))
    reference = f"[{reference_number}] {', '.join(parts)}"
    if metadata.doi:
        reference += f", doi: {metadata.doi}"
    return re.sub(r"\s+", " ", reference).rstrip(".") + "."


def doi_url(document: DocumentRecord) -> str | None:
    metadata = document.citation_metadata
    return f"https://doi.org/{metadata.doi}" if metadata and metadata.doi else None
