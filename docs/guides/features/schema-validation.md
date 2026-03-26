# Schema Validation

A {class}`~dataframely.Schema` class specifies the expected structure and content of a polars DataFrame.
It defines:

- **Columns**: the expected column names, data types, and per-column constraints
- **Rules**: additional row-level or group-level validation expressions

## Columns and column-level rules

Each column in a schema is declared by assigning a {class}`~dataframely.Column` instance to a class attribute:

```python
import dataframely as dy


class UserSchema(dy.Schema):
    id = dy.String(primary_key=True)
    age = dy.UInt8(nullable=False)
    email = dy.String(nullable=True)
```

When validating a DataFrame against this schema, dataframely verifies that:

1. **All expected columns are present** with the correct data types.
2. **Column-level constraints** hold for every row. Common constraints include:
   - `nullable=False` (the default): the column must not contain null values.
   - `primary_key=True`: values in this column (or combination of columns) must be unique.
     See [Primary Keys](primary-keys.md) for details.
   - Type-specific constraints, e.g., `min_length`/`max_length`/`regex` for {class}`~dataframely.String`
     or `min`/`max` for numeric types.

```{note}
Each column type exposes its own set of constraints. Refer to the
{doc}`API reference </api/columns/index>` for a full list.
```

### The `check` parameter

For one-off constraints that do not have a dedicated parameter, every column type accepts a `check`
argument. It receives a polars expression and must return a boolean expression:

```python
class SalarySchema(dy.Schema):
    # Only allow salaries that are a multiple of 500.
    salary = dy.Float64(nullable=False, check=lambda col: col % 500 == 0)
```

Multiple checks can be provided as a list or a dictionary:

```python
class SalarySchema(dy.Schema):
    salary = dy.Float64(
        nullable=False,
        check={
            "multiple_of_500": lambda col: col % 500 == 0,
            "at_least_minimum_wage": lambda col: col >= 1_000,
        },
    )
```

## Schema-level validation rules

Column-level constraints only validate a single column in isolation. When you need to express
constraints that span **multiple columns** or depend on **aggregated values**, use the
{func}`@dy.rule() <dataframely.rule>` decorator:

```python
import polars as pl
import dataframely as dy


class InvoiceSchema(dy.Schema):
    admission_date = dy.Date(nullable=False)
    discharge_date = dy.Date(nullable=False)
    amount = dy.Float64(nullable=False)

    @dy.rule()
    def discharge_after_admission(cls) -> pl.Expr:
        return pl.col("discharge_date") >= pl.col("admission_date")
```

The decorated method receives the schema class as its first argument and must return a polars
`Expr` that evaluates to a **boolean value for every row**. A row is considered valid when the
expression evaluates to `True`.

```{tip}
You can reference a column by its name (e.g. `pl.col("discharge_date")`) or through the schema
attribute (e.g. `InvoiceSchema.discharge_date.col`). The latter is refactoring-safe and allows
IDEs to provide auto-completion.
```

### Group rules

Rules can also be defined on **groups of rows** by passing a `group_by` argument to
{func}`@dy.rule() <dataframely.rule>`. The expression is then evaluated per group and must return
an **aggregated boolean** (one value per group):

```python
class HouseSchema(dy.Schema):
    zip_code = dy.String(nullable=False)
    price = dy.Float64(nullable=False)

    @dy.rule(group_by=["zip_code"])
    def minimum_zip_code_count(cls) -> pl.Expr:
        # Require at least two houses per zip code.
        return pl.len() >= 2
```

All rows belonging to a group that fails a group rule are marked as invalid.

## Schema inheritance

Schemas can be extended through standard Python inheritance. The child schema inherits all columns
and rules from its parent:

```python
class BaseSchema(dy.Schema):
    id = dy.String(primary_key=True)
    created_at = dy.Datetime(nullable=False)


class UserSchema(BaseSchema):
    name = dy.String(nullable=False)
    email = dy.String(nullable=True)
```

`UserSchema.column_names()` returns `["id", "created_at", "name", "email"]`. Inheritance can be
arbitrarily deep and supports multiple inheritance, provided that the same column name is not
defined differently in more than one branch.

## Inspecting a schema

You can inspect a schema by printing it or calling `repr()` on it. This shows all columns together
with their constraints and any custom validation rules:

```python
>>> print(InvoiceSchema)
[Schema "InvoiceSchema"]
  Columns:
    - "admission_date": Date()
    - "discharge_date": Date()
    - "amount": Float64()
  Rules:
    - "discharge_after_admission": [(col("discharge_date")) >= (col("admission_date"))]
```

## Validating data

Once a schema is defined, use {meth}`Schema.validate() <dataframely.Schema.validate>` to check a
DataFrame and raise an error on any violation, or {meth}`Schema.filter() <dataframely.Schema.filter>`
for a "soft" validation that returns both the valid rows and a {class}`~dataframely.FailureInfo`
object describing which rows failed and why.

See the [Quickstart](../quickstart.md) for a step-by-step walkthrough.
