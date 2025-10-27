# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime
import importlib
import inspect
import os
import subprocess
import sys
from subprocess import CalledProcessError
from typing import Any, cast

# -- Project information -----------------------------------------------------

_mod = importlib.import_module("dataframely")


project = "dataframely"
copyright = f"{datetime.date.today().year}, QuantCo, Inc"
author = "QuantCo, Inc."

extensions = [
    # builtin sphinx
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    # external
    "autodocsumm",
    "myst_parser",
    "nbsphinx",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_toolbox.more_autodoc.overloads",
]

## sphinx
# html output
html_theme = "pydata_sphinx_theme"
pygments_style = "lovelace"
html_theme_options = {
    "external_links": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Quantco/dataframely",
            "icon": "fa-brands fa-github",
        },
    ],
}
html_title = "Dataframely"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/favicon.ico"
html_show_sourcelink = False

# markup
default_role = "code"

# object signatures
maximum_signature_line_length = 88

# source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# templating
templates_path = ["_templates"]

## sphinx.ext.autodoc
autoclass_content = "both"
autodoc_default_options = {
    "inherited-members": True,
}

## sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/py-polars/html/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
}

## myst_parser
myst_parser_config = {"myst_enable_extensions": ["rst_eval_roles"]}
nitpick_ignore = [("myst", "group-rules")]

## numpydoc
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

## sphinx_toolbox
overloads_location = ["bottom"]


# Copied and adapted from
# https://github.com/pandas-dev/pandas/blob/4a14d064187367cacab3ff4652a12a0e45d0711b/doc/source/conf.py#L613-L659
# Required configuration function to use sphinx.ext.linkcode
def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Determine the URL corresponding to a given Python object."""
    if domain != "py":
        return None

    module_name = info["module"]
    full_name = info["fullname"]

    _submodule = sys.modules.get(module_name)
    if _submodule is None:
        return None

    _object = _submodule
    for _part in full_name.split("."):
        try:
            _object = getattr(_object, _part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(_object))  # type: ignore
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, line_number = inspect.getsourcelines(_object)
    except OSError:
        line_number = None

    if line_number:
        linespec = f"#L{line_number}-L{line_number + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(cast(str, _mod.__file__)))

    try:
        # See https://stackoverflow.com/a/21901260
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except CalledProcessError:
        # If subprocess returns non-zero exit status
        commit = "main"

    return (
        "https://github.com/quantco/dataframely"
        f"/blob/{commit}/{_mod.__name__.replace('.', '/')}/{fn}{linespec}"
    )


## Hide the signature for classes that should not be instantiated by the user
def hide_class_signature(
    app: Any,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str,
) -> tuple[str, str] | None:
    if what == "class" and (
        name.endswith("FilterResult")
        or name.endswith("FailureInfo")
        or name.endswith("AnnotationImplementationError")
    ):
        # Return empty signature (no args after the class name)
        return "", return_annotation
    # Otherwise, keep default behavior
    return None


def setup(app: Any) -> None:
    app.connect("autodoc-process-signature", hide_class_signature)
