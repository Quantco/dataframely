# Data Generation

Testing data pipelines can be challenging because assessing a pipeline's functionality, performance, or robustness often requires realistic data.
Dataframely supports generating synthetic data that adheres to a schema or a collection of schemas by _sampling_ the schema or collection.
This can make testing considerably easier, for instance, when availability of real data is limited to certain environments, say client infrastructure; or when crafting unit tests that specifically test one edge case which may or may not be present in a real data sample.

## Sampling schemas

### Empty data frames with correct schema

To create an empty data or lazy frame with a valid schema, one can call {meth}`~dataframely.Schema.create_empty` on any schema:

```python
class InvoiceSchema(dy.Schema):
    invoice_id = dy.String(primary_key=True, regex=r"\d{1,10}")
    admission_date = dy.Date(nullable=False)
    discharge_date = dy.Date(nullable=False)
    amount = dy.Decimal(nullable=False)

# Get data frame with correct type hint.
df: dy.DataFrame[InvoiceSchema] = InvoiceSchema.create_empty()
```

While there is technically no data generation involved here, it can still be useful to create empty data frames with correct data types and type hints.

### Generating random data

To generate synthetic (random) data for a schema, one can call {meth}`~dataframely.Schema.sample` on any schema:

```python
class InvoiceSchema(dy.Schema):
    invoice_id = dy.String(primary_key=True, regex=r"\d{1,10}")
    admission_date = dy.Date(nullable=False)
    discharge_date = dy.Date(nullable=False)
    amount = dy.Decimal(nullable=False)

df: dy.DataFrame[InvoiceSchema] = InvoiceSchema.sample(num_rows=100)
```

Note that the data generation also respects per-column validation rules, such as `regex`, `nullable`, or `primary_key`.

### Schema-level validation rules

Dataframely also supports data generation for schemas that include schema-level validation rules:

```python
class InvoiceSchema(dy.Schema):
    invoice_id = dy.String(primary_key=True, regex=r"\d{1,10}")
    admission_date = dy.Date(nullable=False)
    discharge_date = dy.Date(nullable=False)
    amount = dy.Decimal(nullable=False)

    @dy.rule()
    def discharge_after_admission(cls) -> pl.Expr:
        return InvoiceSchema.discharge_date.col >= InvoiceSchema.admission_date.col

# `@dy.rule`s will be respected as well for data generation.
df: dy.DataFrame[InvoiceSchema] = InvoiceSchema.sample(num_rows=100)
```

```{note}
Dataframely will perform "fuzzy sampling" in the presence of custom rules and primary key constraints: it samples in a loop until it finds a data frame of length `num_rows` which adhere to the schema.
The maximum number of sampling rounds is configured via {meth}`~dataframely.Config.set_max_sampling_iterations` and sampling will fail if no valid data frame can be found within this number of rounds.
By fixing this setting to 1, it is only possible to reliably sample from schemas without custom rules and without primary key constraints.
```

### Sampling data frames with specific values

Oftentimes, one may want to sample data for some columns while explicitly specifying values for other columns.
For instance, when writing unit tests for wide data frames with many columns, one is usually only interested in a subset of columns.
Therefore, dataframely provides the `overrides` parameter in {meth}`~dataframely.Schema.sample`, which can be used to manually "override" the values of certain columns while all other columns are sampled as before.
Specifying `overrides` can also be used to allow dataframely to find valid data frames faster,
especially for complex schemas. If `overrides` are specified, `num_rows` can be omitted.

Overrides can be specified in two ways.
The column-wise specification specifies an iterable of values for each specified column:

```python
from datetime import date

# Override values for specific columns.
df: dy.DataFrame[InvoiceSchema] = InvoiceSchema.sample(overrides={
    # Use either <schema>.<column>.name or just the column name as a string.
    InvoiceSchema.invoice_id.name: ["1234567890", "2345678901", "3456789012"],
    # Dataframely will automatically infer the number of rows based on the longest given
    # sequence of values and broadcast all other columns to that shape.
    "admission_date": date(2025, 1, 1),
})
```

The row-wise specification implements an iterable of mappings for the rows that should be sampled. It is particularly helpful if you want to make it easy to understand how values will be combined in specific rows (e.g., when each row represents one object).

```python
from datetime import date
# Override values for specific columns.
df: dy.DataFrame[InvoiceSchema] = InvoiceSchema.sample(overrides=[
    {"invoice_id": "1234567890", "admission_date": date(2025, 1, 1)},
    {"invoice_id": "2345678901", "admission_date": date(2025, 1, 1)},
    {"invoice_id": "3456789012", "admission_date": date(2025, 1, 1)},
])
```

### Providing custom column overrides

Complex validation rules (such as dependencies between columns or ordering criteria) can cause sampling to run excessively long or fail. Specifying additional `overrides` individually in such cases can be tedious. Instead, the {meth}`~dataframely.Schema._sampling_overrides` hook on a schema can be used to specify polars expressions for columns in the schema that will be applied before generated data is filtered during sampling.

```python
import polars as pl
import dataframely as dy
class OrderedSchema(dy.Schema):
    """A schema that requires `iter` to be ordered with respect to `a` and `b`."""

    iter = dy.UInt32(nullable=False)
    a = dy.Int32(nullable=False)
    b = dy.Int32(nullable=False)

    @dy.rule()
    def iter_order_correct(cls) -> pl.Expr:
        return pl.col("iter").rank(method="ordinal") == pl.struct(pl.col("a"), pl.col("b")).rank(method="ordinal")

    @classmethod
    def _sampling_overrides(cls) -> dict[str, pl.Expr]:
        return {
            "iter": pl.struct(pl.col("a"), pl.col("b")).rank(method="ordinal"),
        }

result = OrderedSchema.sample(100)
```

## Sampling collections

Dataframely makes it really easy to set up data for testing an entire relational data model.
Similar to schemas, you can call {meth}`~dataframely.Collection.sample` on any collection.

```python
class DiagnosisSchema(dy.Schema):
    invoice_id = dy.String(primary_key=True)
    code = dy.String(nullable=False, regex=r"[A-Z][0-9]{2,4}")

class HospitalInvoiceData(dy.Collection):
    invoice: dy.LazyFrame[InvoiceSchema]
    diagnosis: dy.LazyFrame[DiagnosisSchema]

invoice_data: HospitalInvoiceData = HospitalInvoiceData.sample(num_rows=10)
```

While this works out of the box for 1:1 relationships between tables, dataframely cannot automatically infer other relations, e.g., 1:N,
that are expressed through `@dy.filter`s in the collection.
Say, for instance, `code` was part of the primary key for `DiagnosisSchema`, and there could be 1 to N diagnoses for an invoice:

```python
class DiagnosisSchema(dy.Schema):
    invoice_id = dy.String(primary_key=True)
    code = dy.String(primary_key=True, regex=r"[A-Z][0-9]{2,4}")

class HospitalInvoiceData(dy.Collection):
    invoice: dy.LazyFrame[InvoiceSchema]
    diagnosis: dy.LazyFrame[DiagnosisSchema]

    @dy.filter()
    def at_least_one_diagnosis(cls) -> pl.Expr:
        return dy.functional.require_relationship_one_to_at_least_one(
            cls.invoice,
            cls.diagnosis,
            on="invoice_id",
        )
```

In this case, calling {meth}`~dataframely.Collection.sample` will fail, because dataframely does not parse the body of `at_least_one_diagnosis` which may contain arbitrary polars expressions.
To address the problem, one can override {meth}`~dataframely.Collection._preprocess_sample` to generate a random number of diagnoses per invoice:

```python
from random import random
from typing import Any, override

from dataframely.random import Generator


class HospitalInvoiceData(dy.Collection):
    invoice: dy.LazyFrame[InvoiceSchema]
    diagnosis: dy.LazyFrame[DiagnosisSchema]

    @dy.filter()
    def at_least_one_diagnosis(cls) -> pl.Expr:
        return dy.functional.require_relationship_one_to_at_least_one(
            cls.invoice,
            cls.diagnosis,
            on="invoice_id",
        )

    @classmethod
    @override
    def _preprocess_sample(cls, sample: dict[str, Any], index: int, generator: Generator):
        # Set common primary key.
        if "invoice_id" not in sample:
            sample["invoice_id"] = str(index)

        # Satisfy filter by adding 1-10 diagnoses.
        if "diagnosis" not in sample:
            # NOTE: Every key in the `sample` corresponds to one member of the collection.
            # In this case, diagnoses contains a list of N diagnoses.
            # Inside the list, one can simply pass empty dictionaries, which means that all columns
            # in the member will be sampled.
            sample["diagnosis"] = [{} for _ in range(0, int(random() * 10) + 1)]
        return sample
```

## Unit testing

To demonstrate the power of data generation for unit testing,
consider the following example.
Here, we want to test a function `get_diabetes_invoice_amounts`:

```python
from polars.testing import assert_frame_equal


class OutputSchema(dy.Schema):
    invoice_id = dy.String(primary_key=True)
    amount = dy.Decimal(nullable=False)


# function under test
def get_diabetes_invoice_amounts(
    invoice_data: HospitalClaims,
) -> dy.LazyFrame[OutputSchema]:
    return OutputSchema.cast(
        invoice_data.diagnosis.filter(DiagnosisSchema.code.col.str.starts_with("E11"))
        .unique(DiagnosisSchema.invoice_id.col)
        .join(invoice_data.invoice, on="invoice_id", how="inner")
    )


# pytest test case
def test_get_diabetes_invoice_amounts() -> None:
    # Arrange
    invoice_data = HospitalInvoiceData.sample(
        overrides=[
            # Invoice with diabetes diagnosis
            {
                "invoice_id": "1",
                "invoice": {"amount": 1500.0},
                "diagnosis": [{"code": "E11.2"}],
            },
            # Invoice without diabetes diagnosis
            {
                "invoice_id": "2",
                "invoice": {"amount": 1000.0},
                "diagnosis": [{"code": "J45.909"}],
            },
        ]
    )
    expected = OutputSchema.validate(
        pl.DataFrame(
            {
                "invoice_id": ["1"],
                "amount": [1500.0],
            }
        ),
        cast=True,
    ).lazy()

    # Act
    actual = get_diabetes_invoice_amounts(invoice_data)

    # Assert
    assert_frame_equal(actual, expected)
```

Dataframely allows us to define test data at the invoice-level, which is easy and intuitive to think about instead of a set of related tables.
Therefore, we can pass a list of dictionaries to `overrides`,
where each dictionary corresponds to an invoice with optional keys per collection member (e.g., `diagnosis`).
The common primary key can be defined as a top-level key in the dictionary and will be transparently added to all members (i.e., `invoice` and `diagnosis`).
Any left out key inside a member will be sampled automatically by dataframely.

````{note}
If you are tired of always providing a key for each member, you can declare any collection member to be "inlined for sampling" using the
{class}`~dataframely.CollectionMember` type annotation:

```python
from typing import Annotated

class HospitalInvoiceData(dy.Collection):
    invoice: Annotated[
        dy.LazyFrame[InvoiceSchema],
        dy.CollectionMember(inline_for_sampling=True),
    ]
    diagnosis: dy.LazyFrame[DiagnosisSchema]
```

This allows you to directly supply non-primary key columns (e.g., `amount`) on the top-level rather than in a subkey with the member's name:

```python
HospitalInvoiceData.sample(overrides=[
    {
        "invoice_id": "1",
        "amount": 1000.0,
        "diagnosis": [{"code": "E11.2"}],
    }
])
```

````

## Customizing data generation

Dataframely allows customizing the data generation process, if the default mechanisms to generate data are not suitable.
To customize the data generation, one can subclass {class}`~dataframely.random.Generator` and override any of the `sample_*` functions.
