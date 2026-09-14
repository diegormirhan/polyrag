"""Puts `backend/` on the import path, for the scripts under tests/ that are run.

pyproject's `pythonpath = ["backend"]` is read by pytest and by nothing else. The
`*_integration.py` files are not pytest tests — they are scripts with a
`__main__` block, and every one of them documents itself as

    uv run python -u tests/test_x_integration.py

which failed at the first `from app...` with ModuleNotFoundError. The command in
the docstring, in the README and in demo/README was wrong for all of them.

Imported for its side effect, before the `app` imports, by each of those scripts.
One module rather than the same three lines copied into seven files. The leading
underscore keeps pytest from collecting it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
