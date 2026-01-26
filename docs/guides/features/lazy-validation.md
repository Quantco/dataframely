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
