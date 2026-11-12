"""The notebooks are code too: every code cell must parse, and every experiment notebook takes the same settings
(papermill injects them into the cell tagged "parameters")."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
SETTINGS = {"MODEL", "DTYPE", "SMOKE", "SEEDS", "N_PER_CELL", "SKIP"}


def _cells(path):
    return json.loads(path.read_text())["cells"]


@pytest.mark.parametrize("path", NOTEBOOKS + [ROOT / "colab.ipynb"], ids=lambda p: p.name)
def test_code_cells_parse(path):
    for c in _cells(path):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_experiment_notebooks_take_the_runner_settings(path):
    params = [c for c in _cells(path) if "parameters" in c["metadata"].get("tags", [])]
    assert len(params) == 1
    names = {t.id for node in ast.walk(ast.parse("".join(params[0]["source"]))) if isinstance(node, ast.Assign)
             for t in node.targets if isinstance(t, ast.Name)}
    assert SETTINGS <= names, SETTINGS - names
