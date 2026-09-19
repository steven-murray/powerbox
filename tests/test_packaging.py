"""Check that the package's runtime imports are declared as dependencies."""

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "powerbox"

# Modules that are only imported when the optional feature that needs them is used.
OPTIONAL_TOPLEVEL = {"pyfftw", "jax"}


def _top_level_third_party_imports(path: Path) -> set[str]:
    """Return the third-party packages imported at module level (not lazily) by a file."""
    tree = ast.parse(path.read_text())
    found = set()
    for node in tree.body:
        # For a try/except, only the `try` body counts: the handlers are fallbacks.
        if isinstance(node, ast.Try):
            nodes = [sub for stmt in node.body for sub in ast.walk(stmt)]
        elif isinstance(node, ast.With | ast.If):
            nodes = ast.walk(node)
        else:
            nodes = [node]
        for sub in nodes:
            if isinstance(sub, ast.Import):
                found.update(alias.name.split(".")[0] for alias in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.level == 0 and sub.module:
                found.add(sub.module.split(".")[0])
    return {name for name in found if name not in sys.stdlib_module_names and name != "powerbox"}


def test_core_runtime_imports_are_declared_dependencies() -> None:
    """A plain ``pip install powerbox`` must be able to ``import powerbox``.

    The ``powerbox.jax`` subpackage is excluded, since it needs the ``jax`` extra.
    """
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {
        re.split(r"[<>=!~;\[ ]", dep, maxsplit=1)[0].lower() for dep in project["dependencies"]
    }

    imported = set()
    for path in SRC.glob("*.py"):
        imported |= _top_level_third_party_imports(path)

    missing = imported - declared - OPTIONAL_TOPLEVEL
    assert not missing, f"imported by powerbox but not in [project].dependencies: {sorted(missing)}"
