---
name: dataframely
description: Best practices for polars data processing with dataframely. Covers definitions of Schema and Collection, usage of
  .validate() and .filter(), type hints, and testing. Use when writing or modifying code involving dataframely or
  polars data frames.
license: BSD-3-Clause
user-invocable: false
---

# Overview

`dataframely` provides two types:

- `dy.Schema` documents and enforces the structure of a single data frame
- `dy.Collection` documents and enforces the relationships between multiple related data frames that each have their
  own `dy.Schema`

## `dy.Schema`

A subclass of `dy.Schema` describes the structure of a single dataframe.

```python
class MyHouseSchema(dy.Schema):
    """A schema for a dataframe describing houses."""

    street = dy.String(primary_key=True)
    number = dy.UInt16(primary_key=True)
    #: Description on the number of rooms.
    rooms = dy.UInt8()
    #: Description on the area of the house.
    area = dy.UInt16()
```

The schema can be used in type hints via `dy.DataFrame[MyHouseSchema]` and `dy.LazyFrame[MyHouseSchema]` to express
schema adherence statically. It can also be used to validate the structure and contents of a data frame at runtime
using validation and filtering.

`dy.DataFrame[...]` and `dy.LazyFrame[...]` are typically referred to as "typed data frames". They are typing-only
wrappers around `pl.DataFrame` and `pl.LazyFrame`, respectively, and only express intent. They are never initialized at
runtime.

### Defining Constraints

Persist all implicit assumptions on the data as constraints in the schema. Use docstrings purely to answer the "what"
about the column contents.

- Use the most specific type possible for each column (e.g. `dy.Enum` instead of `dy.String` when applicable).
- Use pre-defined arguments (e.g. `nullable`, `min`, `regex`) for column-level constraints if possible.
- Use the `check` argument for non-standard column-level constraints that cannot be expressed using pre-defined
  arguments. Prefer the defining the check as a dictionary with keys describing the type of check:

  ```python
  class MySchema(dy.Schema):
      col = dy.UInt8(check={"divisible_by_two": lambda col: (col % 2) == 0})
  ```

- Use rules (i.e. methods decorated with `@dy.rule`) for cross-column constraints. Use expressive names for the rules
  and use `cls` to refer to the schema:

  ```python
  class MySchema(dy.Schema):
      col1 = dy.UInt8()
      col2 = dy.UInt8()

      @dy.rule()
      def col1_greater_col2(cls) -> pl.Expr:
          return cls.col1.col > cls.col2.col
  ```

- Use group rules (i.e. methods decorated with `@dy.rule(group_by=...)`) for cross-row constraints beyond primary key
  checks.

### Referencing Columns

When referencing columns of the schema anywhere in the code, always reference column as attribute of the schema class:

- Use `Schema.column.col` instead of `pl.col("column")` to obtain a `pl.Expr` referencing the column.
- Use `Schema.column.name` to reference the column name as a string.

This allows for easier refactorings and enables lookups on column definitions and constraints via LSP.

## `dy.Collection`

A subclass of `dy.Collection` describes a set of related data frames, each described by a `dy.Schema`. Data frames in a
collection should share at least a subset of their primary key.

```python
class MyStreetSchema(dy.Schema):
    """A schema for a dataframe describing streets."""

    # Shared primary key component with MyHouseSchema
    street = dy.String(primary_key=True)
    city = dy.String()


class MyCollection(dy.Collection):
    """A collection of related dataframes."""

    houses: dy.LazyFrame[MyHouseSchema]
    streets: dy.LazyFrame[MyStreetSchema]
```

The collection can be used in a standalone manner (much like a dataclass). It can also be used to validate the
structure and contents of its members and their relationships at runtime using validation and filtering.

### Defining Constraints

Persist all implicit assumptions about the relationships between the collections' data frames as constraints in the
collection.

- Use filters (i.e. methods decorated with `@dy.filter`) to enforce assumptions about the relationships (e.g. 1:1, 1:N)
  between the collections' data frames. Leverage `dy.functional` for writing filter logic.

  ```python
  class MyCollection(dy.Collection):
      houses: dy.LazyFrame[MyHouseSchema]
      streets: dy.LazyFrame[MyStreetSchema]

      @dy.filter()
      def all_houses_on_known_streets(cls) -> pl.LazyFrame:
          return dy.functional.require_relationship_one_to_at_least_one(
              cls.streets, cls.houses, on="street"
          )
  ```

# Usage Conventions

## Clear Interfaces

Structure data processing code with clear interfaces documented using `dataframely` type hints:

```python
def preprocess(raw: dy.LazyFrame[MyRawSchema]) -> dy.DataFrame[MyPreprocessedSchema]:
    # Internal data frames do not require schemas
    df: pl.LazyFrame = ...
    return MyPreprocessedSchema.validate(df, cast=True)
```

- Use schemas for all input and output data frames in a function. Omit type hints if the function is a private helper
  (prefixed with `_`) unless the schema critically improves readability or testability.
- Omit schemas for short-lived temporary data frames. Never define schemas for function-local data frames.

## Validation and Filtering

Both `.validate` and `.filter` enforce the schema at runtime. Pass `cast=True` for safe type-casting.

- **`Schema.validate`** — raises on failure. Use when failures are unexpected (e.g. transforming already-validated
  data).
- **`Schema.filter`** — returns valid rows plus a `FailureInfo` describing filtered-out rows. Use when failures are
  possible and should be handled gracefully. Failures should either be kept around or logged for introspection. The
  `FailureInfo` object provides several utility methods to obtain information about the failures:
  - `len(failure)` provides the total number of failures
  - `failure.counts()` provides the number of violations by rule
  - `failure.invalid()` provides the data frame of invalid rows
  - `failure.details()` provides the data frame of invalid rows with additional columns providing information on which
    rules were violated

When performing validation or filtering, prefer using `pipe` to clarify the flow of data:

```python
result = df.pipe(MySchema.validate)
out, failures = df.pipe(MySchema.filter)
```

### Pure Casting

Use `Schema.cast` as an escape-hatch when it is already known that the data frame conforms to the schema and the
runtime cost of the validation should not be incurred. Generally, prefer using `Schema.validate` or `Schema.filter`.

## Testing

Unless otherwise specified by the user or the project context, add unit tests for all (non-private) methods performing
data transformations.

- Do not test properties already guaranteed by the schema (e.g. data types, nullability, value constraints).

### Test structure

Write tests with the following structure:

1. "Arrange": Define synthetic input data and expected output
2. "Act": Execute the transformation
3. "Assert": Compare expected and actual output using `assert_frame_equal` from `polars.testing`

```python
from polars.testing import assert_frame_equal


def test_grouped_sum():
    df = pl.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "a", "b"],
    }).pipe(MyInputSchema.validate, cast=True)

    expected = pl.DataFrame({
        "col1": ["a", "b"],
        "col2": [3, 3],
    })

    result = my_code(df)

    assert_frame_equal(expected, result)
```

### Generating Synthetic Test Data

Use `dataframely`'s synthetic data generation for creating inputs to functions requiring typed data frames in their
input. Generate synthetic data for schemas as follows:

- Use `MySchema.sample(num_rows=...)` to generate fully random data when exact contents don't matter.
- Use `MySchema.sample(overrides=...)` to generate random data with specific columns pinned to certain values for
  testing specific functionality. Prefer using dicts of lists for overrides unless specifically prompted otherwise.
  - When using dicts of lists: for providing overrides that are constant across all rows, provide scalar values instead
    of lists of equal values.
- Always use `MySchema.create_empty()` instead of sampling with empty overrides when an empty data frame is needed.

Synthetic data for collections should be generated as follows:

- Use `MyCollection.sample(num_rows=...)` to generate fully random data when exact contents don't matter.
- Use `MyCollection.sample(overrides=...)` to generate random data where certain values of the collection members
  matter. Use lists of dicts for providing overrides as "objects" spanning the collection members.
  - Values for shared primary keys must be provided at the root of the dictionaries
  - Values for individual collection members must be provided in nested dictionaries under the keys corresponding to
    the collection member names.
- Always use `MyCollection.create_empty()` instead of sampling with empty overrides when an empty collection is needed.

## I/O Conventions

When writing typed data frames to disk, prefer using `MySchema.write_...` instead of using `write_...` directly on the
data frame. This ensures that schema metadata is persisted alongside the data and can be leveraged when reading the
data back in.

When reading typed data frames from disk, prefer using `MySchema.read_...` instead of using `pl.read_...` directly from

# Getting more information

`dataframely` provides clear function signatures, type hints and docstrings for the full public API. For more
information, inspect the source code in the site packages. If available, always use the LSP tool to find documentation.
