"""
paths.py
========
Bootstrap import paths for scripts under retriever/.
"""

from __future__ import annotations

import sys
from pathlib import Path

RETRIEVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RETRIEVER_DIR.parent


def setup_import_paths() -> Path:
    """Add project root and retriever/ to sys.path. Returns PROJECT_ROOT."""
    for entry in (str(PROJECT_ROOT), str(RETRIEVER_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return PROJECT_ROOT
