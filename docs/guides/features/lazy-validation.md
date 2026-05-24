# Lazy Validation

In many cases, dataframely's capability to validate and filter input data is used at core application boundaries.
As a result, `validate` and `filter` are generally expected to be used at points where `collect` is called on a lazy
frame. However, there may be situations where validation or filtering should simply be added to the lazy computation
graph. Starting in dataframely v2, this is supported via a custom polars plugin.

## The `eager` parameter

All of the following methods expose an `eager: bool` parameter:

- {meth}`Schema.validate() <dataframely.Schema.validate>`
- {meth}`Schema.filter() <dataframely.Schema.filter>`
- {meth}`Collection.validate() <dataframely.Collection.validate>`
- {meth}`Collection.filter() <dataframely.Collection.filter>`

By default, `eager=True`. However, users may decide to set `eager=False` in order to simply append the validation or
the filtering operation to the lazy frame. For example, one might decide to run validation lazily:

```python
def validate_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.pipe(MySchema.validate, eager=False)
```

When `eager=False`, validation is only run once the lazy frame is collected. If input data does not satisfy the schema,
no error is raised here, yet.

## Error Types

Due to current limitations in polars plugins, the type of error that is being raised from the `validate` function (both
for schemas and collections) is dependent on the value of the `eager` parameter:

- When `eager=True`, a {class}`~dataframely.ValidationError` is raised from the `validate` function
- When `eager=False`, a {class}`~polars.exceptions.ComputeError` is raised from the `collect` function

```{note}
For schemas, the error _message_ itself is equivalent.
For collections, the error message for `eager=False` is limited and non-deterministic: the error message only includes
information about a single member and, if multiple members fail validation, the member that the error message refers to
may vary across executions.
```

```{note}
When the lazy frame is collected on the polars _streaming_ engine, lazy validation may not surface _all_ validation
issues: validation is aborted as soon as the first failure is encountered. As a result, both the set of rules reported
in the error message and the specific failure surfaced may be non-deterministic across executions.
```

## Including failure examples in error messages

By default, validation error messages report only the name of each failing rule and the number of rows that violated it.
For easier debugging, dataframely can additionally include a few example rows for each failing rule. This is configured
via {meth}`~dataframely.Config.set_max_failure_examples` (or the `max_failure_examples` keyword on the
{class}`~dataframely.Config` context manager) and applies to both `eager=True` and `eager=False`:

```python
import dataframely as dy

with dy.Config(max_failure_examples=5):
    MySchema.validate(df)
```

For column-level rules, examples include the value in the offending column. For schema-level rules, examples include all
data columns of the schema, except for the `primary_key` rule where examples are limited to the primary key columns.
The default value of `0` disables examples entirely.
