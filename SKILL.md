---
name: dataframely
description: A declarative, Polars-native data frame validation library. Use when implementing data processing logic in polars.
license: BSD-3-Clause
---

# Dataframely skill

`dataframely` provides `dy.Schema` and `dy.Collection` to document and enforce the structure of single or multiple
related data frames.

## `dy.Schema` example

A `dy.Schema` describes the structure of a single dataframe.

```python
class HouseSchema(dy.Schema):
    """A schema for a dataframe describing houses."""

    street: dy.String(primary_key=True)
    number: dy.UInt16(primary_key=True)
    # Number of rooms
    rooms: dy.UInt8()
    # Area in square meters
    area: dy.UInt16()
```

## `dy.Collection` example

A `dy.Collection` describes a set of related dataframes, each described by a `dy.Schema`. Dataframes in a collection
should share at least a subset of their primary key.

```python
class MyStreetSchema(dy.Schema):
    """A schema for a dataframe describing streets."""

    # Shared primary key component with MyHouseSchema
    street: dy.String(primary_key=True)
    city: dy.String()


class MyCollection(dy.Collection):
    """A collection of related dataframes."""

    houses: MyHouseSchema
    streets: MyStreetSchema
```

# Usage conventions

## Use clear interfaces

Structure data processing code with clear interfaces documented using `dataframely` type hints:

```python
def preprocess(raw: dy.LazyFrame[MyRawSchema]) -> dy.DataFrame[MyPreprocessedSchema]:
    # Internal dataframes do not require schemas
    df: pl.LazyFrame = ...
    return MyPreprocessedSchema.validate(df, cast=True)
```

Use schemas for all input, output, and intermediate dataframes. Schemas may be omitted for short-lived temporary
dataframes and private helper functions (prefixed with `_`).

## `filter` vs `validate`

Both `.validate` and `.filter` enforce the schema at runtime. Pass `cast=True` for safe type-casting.

- **`Schema.validate`** — raises on failure. Use when failures are unexpected (e.g. transforming already-validated
  data).
- **`Schema.filter`** — returns valid rows plus a `FailureInfo` describing filtered-out rows. Use when failures are
  possible and should be handled gracefully (e.g. logging and skipping invalid rows).

## Testing

Every data transformation must have unit tests. Test each branch of the transformation logic. Do not test properties
already guaranteed by the schema.

### Test structure

1. Create synthetic input data
2. Define the expected output
3. Execute the transformation
4. Compare using `assert_frame_equal` from `polars.testing` (or `diffly.testing` if installed)

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

    assert assert_frame_equal(expected, result)
```

### Generating synthetic input data

For complex schemas where only some columns are relevant to the test, use `dataframely`'s synthetic data generation:

```python
# Random data meeting all schema constraints
random_data = MyInputSchema.sample(num_rows=100)
```

Use fully random data for property tests where exact contents don't matter. Use overrides to pin specific columns while
randomly sampling the rest:

```python
random_data_with_overrides = HouseSchema.sample(
    num_rows=5,
    overrides={
        "street": ["Main St.", "Main St.", "Main St.", "Second St.", "Second St."],
    }
)
```

# Getting more information

`dataframely` relies on clear function signatures, type hints and doc strings. If you need more information, check the
locally installed code.
