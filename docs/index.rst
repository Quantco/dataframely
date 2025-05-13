Dataframely
============

Dataframely is a Python package to validate the schema and content of `polars <https://pola.rs/>`_ data frames.
Its purpose is to make data pipelines more robust by ensuring that data meet expectations and more readable by adding schema information to data frame type hints.

Features
--------

- Declaratively define schemas as classes with arbitrary inheritance structure
- Specify column-specific validation rules (e.g. nullability, minimum string length, ...)
- Specify cross-column and group validation rules with built-in support for checking the primary key property of a column set
- Specify validation constraints across collections of interdependent data frames
- Validate data frames softly by simply filtering out rows violating rules instead of failing hard
- Introspect validation failure information for run-time failures
- Enhanced type hints for validated data frames allowing users to clearly express expectations about inputs and outputs (i.e., contracts) in data pipelines
- Integrate schemas with external tools (e.g., ``sqlalchemy`` or ``pyarrow``)
- Generate test data that comply with a schema or collection of schemas and its validation rules

Contents
========

.. toctree::
    :caption: Contents
    :maxdepth: 1

    Installation <sites/installation.rst>
    Quickstart <sites/quickstart.rst>
    Real-world Example <sites/examples/real-world.ipynb>
    FAQ <sites/faq.rst>
    Development Guide <sites/development.rst>
    Versioning <sites/versioning.rst>

API Documentation
=================

.. toctree::
    :caption: API Documentation
    :maxdepth: 1

    Collection <_api/dataframely.collection>
    Column Types <_api/dataframely.columns>
    Config <_api/dataframely.config>
    Random Data Generation <_api/dataframely.random>
    Failure Information <_api/dataframely.failure>
    Schema <_api/dataframely.schema>
