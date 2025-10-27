# Dataframely

Dataframely is a Python package to validate the schema and content of [polars](https://pola.rs/) data frames.
Its purpose is to make data pipelines more robust by ensuring that data meet expectations and more readable by adding
schema information to data frame type hints.

## Features

- Declaratively define schemas as classes with arbitrary inheritance structure
- Specify column-specific validation rules (e.g. nullability, minimum string length, ...)
- Specify cross-column and group validation rules with built-in support for checking the primary key property of a
  column set
- Specify validation constraints across collections of interdependent data frames
- Validate data frames softly by simply filtering out rows violating rules instead of failing hard
- Introspect validation failure information for run-time failures
- Enhanced type hints for validated data frames allowing users to clearly express expectations about inputs and
  outputs (i.e., contracts) in data pipelines
- Integrate schemas with external tools (e.g., `sqlalchemy` or `pyarrow`)
- Generate test data that comply with a schema or collection of schemas and its validation rules

```{toctree}
:maxdepth: 2
:hidden:

guides/index
api/index
```
