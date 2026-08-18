"""Keep the deliberately small UI on the HTTP side of the architecture."""

from __future__ import annotations

import ast
from pathlib import Path


def test_streamlit_ui_does_not_import_project_internals() -> None:
    source = Path("apps/streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any(name.startswith("adaspilot") for name in imported)
    assert "httpx" in imported
