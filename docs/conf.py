"""Sphinx configuration for the trajopt documentation."""

import shutil
from pathlib import Path

# -- Example notebooks -------------------------------------------------------
# Sphinx can only include documents inside its source tree, so copy the
# example notebooks from examples/ into docs/examples/notebooks/ at build time.
# The copied tree is git-ignored; examples/ remains the single source of truth.

_DOCS_DIR = Path(__file__).parent
_EXAMPLES_SRC = _DOCS_DIR.parent / "examples"
_NOTEBOOKS_DST = _DOCS_DIR / "examples" / "notebooks"

if _NOTEBOOKS_DST.exists():
    shutil.rmtree(_NOTEBOOKS_DST)
for _nb in sorted(_EXAMPLES_SRC.glob("*/*.ipynb")):
    if ".ipynb_checkpoints" in _nb.parts:
        continue
    _dst = _NOTEBOOKS_DST / _nb.parent.name / _nb.name
    _dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_nb, _dst)

# -- Project information -----------------------------------------------------

project = "trajopt"
author = "Skye Mceowen"
release = "0.0.1"
copyright = "2026, Skye Mceowen"

# -- General configuration ---------------------------------------------------

extensions = [
    "autoapi.extension",
    "myst_nb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- sphinx-autoapi ----------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = ["../src/trajopt"]
autoapi_root = "autoapi"
autoapi_add_toctree_entry = True
autoapi_ignore = ["*__pycache__*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

# -- myst-nb -----------------------------------------------------------------

# Notebook outputs are not committed and not final; render without executing.
nb_execution_mode = "off"

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "trajopt"
