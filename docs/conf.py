# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import datetime
import importlib
import inspect
import os
import subprocess
import sys
from subprocess import CalledProcessError
from typing import cast

_mod = importlib.import_module("dataframely")


project = "dataframely"
copyright = f"{datetime.date.today().year}, QuantCo, Inc"
author = "QuantCo, Inc."

extensions = [
    "nbsphinx",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinxcontrib.apidoc",
]

numpydoc_class_members_toctree = False

apidoc_module_dir = "../dataframely"
apidoc_output_dir = "_api"
apidoc_module_first = True
apidoc_extra_args = ["--implicit-namespaces"]

autodoc_default_options = {
    "inherited-members": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = "Dataframely"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.ico"


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
