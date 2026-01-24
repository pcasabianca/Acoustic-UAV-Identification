"""Pytest conftest to ensure local visualization tools are importable.

This prepends the `6 - Visualization` directory to `sys.path` so tests
can `import tools.*` from the local toolkit without relying on a
top-level package shim.
"""
import sys
from pathlib import Path

_viz_base = Path(__file__).resolve().parents[1]
_repo_root = _viz_base.parent
# Ensure viz base first so local `tools` package is preferred, then repo root for `config` and other top-level modules.
sys.path.insert(0, str(_viz_base))
sys.path.insert(1, str(_repo_root))
