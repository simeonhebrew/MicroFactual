# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "MicroFactual"
copyright = "2025, Simeon Hebrew, Lawrence Adu-Gyamfi"
author = "Simeon Hebrew, Lawrence Adu-Gyamfi"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
    # Don't pull in inherited sklearn base-class methods — their docstrings
    # reference sklearn-internal labels that don't resolve here (noise).
    "inherited-members": False,
}

# Suppress cross-reference warnings that originate in third-party (sklearn)
# docstrings — e.g. the auto-injected metadata-routing methods — and harmless
# duplicate-object refs from package re-exports. Our own .rst uses none of these
# (no ``:ref:`` labels or glossary terms), so this only silences sklearn noise.
suppress_warnings = ["ref.python", "ref.ref", "ref.term"]
nitpicky = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

# Optional: Use NumPy style docstrings for best rendering
napoleon_google_docstring = True
# napoleon_numpy_docstring = True
