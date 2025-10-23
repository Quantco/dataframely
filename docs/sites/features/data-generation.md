# Data Generation

Testing data pipelines can be challenging because assessing a pipeline's functionality, performance, or robustness often requires realistic data.
Dataframely supports generating synthetic data that adheres to a schema or a collection of schemas.
This can make testing considerably easier, for instance, when availability of real data is limited to certain environments, say client infrastructure; or when crafting unit tests that specifically test one edge case which may or may not be present in a real data sample.

## Empty data frames with correct schema

To create an empty data frame with a valid schema, one can call {meth}`~dataframely.Schema.create_empty` on any schema:

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

## Generating random data

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
    def discharge_after_admission() -> pl.Expr:
        return InvoiceSchema.discharge_date.col >= InvoiceSchema.admission_date.col

# `@dy.rule`s will be respected as well for data generation.
df: dy.DataFrame[InvoiceSchema] = InvoiceSchema.sample(num_rows=100)
```

```{note}
Dataframely will perform "fuzzy sampling" in the presence of custom rules and primary key constraints: it samples in a loop until it finds a data frame of length `num_rows` which adhere to the schema.
The maximum number of sampling rounds is configured via {meth}`~dataframely.Config.set_max_sampling_iterations`.
By fixing this setting to 1, it is only possible to reliably sample from schemas without custom rules and without primary key constraints.
```

### Overriding/excluding specific values during sampling

Oftentimes, one may want to sample data for some columns while explicitly specifying values for other columns.
For instance, when writing unit tests for wide data frames with many columns, one is usually only interested in a subset of columns.
Therefore, dataframely provides the `overrides` parameter in {meth}`~dataframely.Schema.sample`, which can be used to manually "override" the values of certain columns while all other columns are sampled as before.

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

### Data generation for collections

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

### Customizing data generation

Dataframely allows customizing the data generation process, if the default mechanisms to generate data are not suitable.
To customize the data generation, one can subclass {class}`~dataframely.random.Generator` and override any of the `sample_*` functions.

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
    invoice_data = HospitalClaims.sample(
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
