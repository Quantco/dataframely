# Schema Validation

Dataframely validates a data frame against a {class}`~dataframely.Schema` by evaluating
every validation rule as a native polars expression.
This guide explains the validation pipeline in detail, including the different validation
methods, how rules are applied, and how to introspect failures.

## How validation rules are applied

When a data frame is validated, dataframely collects the following rules and evaluates them
in a single polars pass:

1. **Column-level rules** — Automatically derived from the parameters of each column
   definition (e.g. `nullable`, `min_length`, `regex`, …). These rules are named
   `<column>|<rule>`, e.g. `zip_code|min_length` or `num_bedrooms|nullability`.

2. **Primary key rule** — If any column in the schema has `primary_key=True`, dataframely
   automatically adds a `primary_key` rule that checks for uniqueness across all primary
   key columns.

3. **Custom schema-level rules** — Rules defined on the schema class using the
   {func}`~dataframely.rule` decorator. These rules may reference any column in the schema
   and can also aggregate across rows when a `group_by` parameter is provided.

For every row the combined result of all rules determines whether the row is valid.
A row is considered valid only when every applicable rule evaluates to `True`.

## Validation methods

Dataframely exposes three methods for checking a data frame against a schema.

### `validate` — strict validation

{meth}`Schema.validate() <dataframely.Schema.validate>` raises an exception if any row
fails validation:

```python
import polars as pl
import dataframely as dy


class HouseSchema(dy.Schema):
    zip_code = dy.String(nullable=False, min_length=3)
    num_bedrooms = dy.UInt8(nullable=False)
    num_bathrooms = dy.UInt8(nullable=False)
    price = dy.Float64(nullable=False)

    @dy.rule()
    def reasonable_bathroom_to_bedroom_ratio(cls) -> pl.Expr:
        ratio = pl.col("num_bathrooms") / pl.col("num_bedrooms")
        return (ratio >= 1 / 3) & (ratio <= 3)


df = pl.DataFrame({
    "zip_code": ["01234", "01234", "1", "213", "123", "213"],
    "num_bedrooms": [2, 2, 1, None, None, 2],
    "num_bathrooms": [1, 2, 1, 1, 0, 8],
    "price": [100_000, 110_000, 50_000, 80_000, 60_000, 160_000],
})

validated_df: dy.DataFrame[HouseSchema] = HouseSchema.validate(df, cast=True)
```

If any row is invalid, a {class}`~dataframely.ValidationError` is raised with a summary of
which rules failed and how many rows were affected:

```
ValidationError: 2 rules failed validation:
* Column 'num_bedrooms' failed validation for 1 rules:
- 'nullability' failed for 2 rows
* Column 'zip_code' failed validation for 1 rules:
- 'min_length' failed for 1 rows
```

On success, `validate` returns a data frame with the type hint
`dy.DataFrame[HouseSchema]`, indicating that the data has been validated.
Columns not defined in the schema are silently dropped.

### `filter` — soft validation

{meth}`Schema.filter() <dataframely.Schema.filter>` never raises a
{class}`~dataframely.ValidationError`.
Instead, it returns a tuple of the valid rows and a
{class}`~dataframely.FailureInfo` object carrying details about the rows that failed:

```python
valid_df, failure = HouseSchema.filter(df, cast=True)
```

`valid_df` is a `dy.DataFrame[HouseSchema]` containing only the rows that passed every
rule.
The `failure` object lets you inspect why the remaining rows were rejected (see
[Inspecting failures](#inspecting-failures) below).

### `is_valid` — boolean check

{meth}`Schema.is_valid() <dataframely.Schema.is_valid>` is a convenience method that
returns `True` when all rows pass all rules, and `False` otherwise.
It never raises a {class}`~dataframely.ValidationError`:

```python
if not HouseSchema.is_valid(df, cast=True):
    print("Data does not satisfy HouseSchema")
```

## Structural errors vs. data errors

Dataframely distinguishes two categories of errors:

- **{class}`~dataframely.SchemaError`** — raised when the *structure* of the data frame
  does not match the schema, i.e., a required column is missing or a column has the wrong
  data type and `cast=False`.
  A `SchemaError` is raised by all three validation methods.

- **{class}`~dataframely.ValidationError`** — raised by `validate` when the *content*
  of the data frame violates at least one rule.

`is_valid` catches both types of errors and returns `False` instead of raising.

## Type casting during validation

All three methods accept a `cast` keyword argument.
When `cast=True`, dataframely attempts to cast each column to the dtype defined in the
schema before running the validation rules.

```python
df_raw = pl.DataFrame({
    "zip_code": ["01234", "67890"],
    "num_bedrooms": [2, 3],
    "num_bathrooms": [1, 1],
    "price": ["100000", "200000"],  # stored as strings
})

# Cast 'price' from String to Float64 before validating
validated_df = HouseSchema.validate(df_raw, cast=True)
```

If a cast fails for a particular row (e.g. a string value cannot be converted to a
number), that row is treated as invalid and included in the `FailureInfo` with the rule
name `<column>|dtype`.

## Inspecting failures

The {class}`~dataframely.FailureInfo` object returned by `filter` provides several ways
to understand which rows failed and why.

### `counts` — per-rule failure counts

```python
valid_df, failure = HouseSchema.filter(df, cast=True)
print(failure.counts())
```

Returns a dictionary mapping each rule that was violated to the number of rows that
failed it:

```python
{
    "reasonable_bathroom_to_bedroom_ratio": 1,
    "zip_code|min_length": 1,
    "num_bedrooms|nullability": 2,
}
```

### `invalid` — the failing rows

```python
failed_rows = failure.invalid()
```

Returns a plain `pl.DataFrame` containing the original data for each row that failed
at least one rule.

### `details` — per-row per-rule breakdown

```python
details = failure.details()
```

Returns the same rows as `invalid` but augmented with one additional column per rule,
with values `"valid"`, `"invalid"`, or `"unknown"`:

| zip_code | num_bedrooms | … | reasonable_bathroom_to_bedroom_ratio | zip_code\|min_length | num_bedrooms\|nullability |
|----------|-------------|---|--------------------------------------|----------------------|--------------------------|
| 1        | 1           | … | valid                                | invalid              | valid                    |
| 213      | null        | … | valid                                | valid                | invalid                  |
| 213      | 2           | … | invalid                              | valid                | valid                    |

A value of `"unknown"` is reported when a rule could not be evaluated reliably, which
can happen when `cast=True` and dtype casting fails for a value in that row.

### `cooccurrence_counts` — co-occurring rule failures

```python
cooccurrences = failure.cooccurrence_counts()
```

Returns a mapping from *sets* of rules to the number of rows where exactly those rules
all failed together.
This is useful for understanding whether certain rules tend to fail in combination.

## Superfluous columns

When validating, any column in the input data frame that is not defined in the schema is
silently dropped from the output of `validate` and `filter`.
This keeps the output predictable regardless of the input shape.
