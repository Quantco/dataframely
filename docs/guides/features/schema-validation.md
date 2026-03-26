# Schema Validation

`dataframely` validates data frames against a {class}`~dataframely.Schema` in two complementary ways: checking the
*structure* of the data frame (column names and data types) and checking the *contents* (the actual values in each row).

## The Validation Pipeline

Whenever `validate` or `filter` is called on a schema, the following steps are executed:

1. **Schema matching**: `dataframely` checks that all columns defined in the schema are present in the data frame with
   the correct data types. Missing columns or type mismatches raise a {class}`~dataframely.exc.SchemaError`.
2. **Rule evaluation**: Each rule (see below) is evaluated against every row of the data frame, producing a boolean
   column for each rule. A value of `True` means the row is valid with respect to that rule; `False` means it is not.
3. **Aggregation**: A row is considered valid if and only if *all* rules evaluate to `True` for it.

The `cast=True` option adds an optional pre-processing step before schema matching: it attempts to cast each column to
the data type required by the schema. If a value cannot be cast, that row is treated as invalid.

## Types of Validation Rules

### Column-level rules

Every column type automatically generates rules based on its parameters:

- **Nullability** (`nullable=False`, the default): a `nullability` rule checks that no value in the column is `null`.
- **Type-specific constraints**: each column type exposes parameters for common constraints such as `min`/`max` for
  numeric types, `min_length`/`max_length`/`regex` for strings, etc. Each of these parameters translates to its own
  named rule.

These rules are named `<column>|<rule>`, for example `zip_code|min_length`.

You can also attach arbitrary inline validation logic to any column using the `check` parameter, which accepts a
callable, a list of callables, or a dictionary mapping names to callables:

```python
class TemperatureSchema(dy.Schema):
    celsius = dy.Float64(check=lambda col: col > -273.15)
```

### Primary key rules

When at least one column is marked as `primary_key=True`, `dataframely` automatically adds a rule named `primary_key`
that checks for uniqueness across the combination of all primary key columns.

### Schema-level rules

For constraints that span multiple columns or that must be evaluated across groups of rows, you can define custom rules
on the schema class using the {func}`~dataframely.rule` decorator:

```python
class HouseSchema(dy.Schema):
    num_bedrooms = dy.UInt8(nullable=False)
    num_bathrooms = dy.UInt8(nullable=False)

    @dy.rule()
    def reasonable_bathroom_to_bedroom_ratio(cls) -> pl.Expr:
        ratio = pl.col("num_bathrooms") / pl.col("num_bedrooms")
        return (ratio >= 1 / 3) & (ratio <= 3)
```

The rule function must return a polars expression that yields one boolean value per row. `True` means the row is valid.

#### Group rules

Rules can also be evaluated on *groups* of rows using the `group_by` parameter of {func}`~dataframely.rule`. Inside a
group rule the expression must aggregate to a single boolean value per group:

```python
class HouseSchema(dy.Schema):
    zip_code = dy.String(nullable=False)

    @dy.rule(group_by=["zip_code"])
    def minimum_zip_code_count(cls) -> pl.Expr:
        return pl.len() >= 2
```

The aggregated result is broadcast back to every row within the group before the global pass/fail decision is made.

## Null Handling in Rules

By default, if a rule expression evaluates to `null` for a row (for example, because one of the columns it references
contains a null value), `dataframely` treats that row as *valid* with respect to that rule. Nullability is checked
separately by the `nullability` rule, so this convention avoids double-counting failures.

```{note}
You can override this behavior explicitly by handling `null` values in the rule expression itself, e.g. using
`pl.Expr.fill_null(False)`.
```

## Validation Methods

`dataframely` exposes four methods related to validation:

| Method | Returns | Raises on failure |
|---|---|---|
| {meth}`~dataframely.Schema.validate` | Validated data frame | {class}`~dataframely.ValidationError` |
| {meth}`~dataframely.Schema.filter` | `(valid_rows, FailureInfo)` | Never (invalid rows are filtered out) |
| {meth}`~dataframely.Schema.is_valid` | `bool` | Never |
| {meth}`~dataframely.Schema.cast` | Data frame (no content check) | Never |

**`validate`** is the strictest method: it raises a {class}`~dataframely.ValidationError` if *any* row in the data
frame violates a rule. Use it when you expect all data to be valid and want to fail fast.

**`filter`** performs *soft-validation*: invalid rows are silently removed and returned via a
{class}`~dataframely.FailureInfo` object that lets you inspect which rules were violated and by how many rows. Use it
when you want to salvage the valid portion of a data frame while still tracking failures.

**`is_valid`** provides a simple boolean answer. It never raises and is useful for conditional logic or quick
sanity checks.

**`cast`** performs only a dtype cast and column selection — it does **not** check the contents of the data frame at
all. Use it only when you already know the data is valid.

```{note}
`validate` is implemented on top of `filter`: it calls `filter` internally and raises a
{class}`~dataframely.ValidationError` when `filter` reports any failures.
```
