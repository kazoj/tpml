"""Charge les fonctions du notebook pour les tester sans dupliquer le code.

Le notebook `projet.ipynb` reste la seule source : on en extrait (avec `ast`) les imports,
les constantes de configuration et les fonctions testées, puis on les exécute dans un
espace de noms. Les cellules d'analyse (calculs longs, figures) ne sont pas exécutées.
"""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

CONSTANTS = {"SEED", "DATA_DIR", "FIG_DIR", "KINDS", "LABELS", "CV", "SCORING"}
FUNCTIONS = {
    "load_raw", "clean", "load_clean", "split", "load_split", "make_preprocessor",
    "cv_scores", "iqr_outliers", "evaluate", "best_f1_threshold",
}


def _keep(node):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.FunctionDef):
        return node.name in FUNCTIONS
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets)
    return False


@pytest.fixture(scope="session")
def nb():
    """Fonctions et constantes du notebook, exécutées depuis la racine du projet (chemins relatifs)."""
    cells = json.loads((ROOT / "projet.ipynb").read_text(encoding="utf-8"))["cells"]
    source = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    module = ast.Module(body=[n for n in ast.parse(source).body if _keep(n)], type_ignores=[])

    namespace = {}
    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        exec(compile(module, "projet.ipynb", "exec"), namespace)
        missing = FUNCTIONS - namespace.keys()
        assert not missing, f"fonctions introuvables dans le notebook : {missing}"
        yield SimpleNamespace(**namespace)
    finally:
        os.chdir(previous)
