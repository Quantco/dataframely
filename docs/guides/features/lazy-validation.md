# Lazy Validation

In many cases, dataframely's capability to validate and filter input data is used at core application boundaries.
As a result, `validate` and `filter` are generally expected to be used at points where `collect` is called on a lazy
frame. However, there may be situations where validation or filtering should simply be added to the lazy computation
graph. This is supported via a custom polars plugin.

## Schemas: lazy-in-lazy-out

{meth}`Schema.validate() <dataframely.Schema.validate>` and {meth}`Schema.filter() <dataframely.Schema.filter>` follow
the same "eager-in-eager-out" and "lazy-in-lazy-out" semantics as polars: if a {class}`~polars.DataFrame` is passed,
validation and filtering are performed immediately. If a {class}`~polars.LazyFrame` is passed, the validation or
filtering operation is simply appended to the lazy frame:

```python
def validate_lf(lf: pl.LazyFrame) -> dy.LazyFrame[MySchema]:
    return lf.pipe(MySchema.validate)
```

For lazy inputs, validation is only run once the lazy frame is collected. If input data does not satisfy the schema,
no error is raised here, yet.

## Collections: the `lazy` parameter

{meth}`Collection.validate() <dataframely.Collection.validate>` and
{meth}`Collection.filter() <dataframely.Collection.filter>` collect their results by default, regardless of whether the
members are passed as data frames or lazy frames. Set `lazy=True` to keep the results lazy:

```python
collection = MyCollection.validate(data, lazy=True)
```

Note that `lazy=True` cannot be used on a collection that defines at least one eager member, i.e. a member annotated
with `dy.DataFrame`.

## Error Types

Due to current limitations in polars plugins, the type of error that is being raised from the `validate` function (both
for schemas and collections) is dependent on whether validation is performed eagerly or lazily:

- When validating eagerly, a {class}`~dataframely.exc.ValidationError` is raised from the `validate` function
- When validating lazily, a {class}`~polars.exceptions.ComputeError` is raised from the `collect` function

```{note}
For schemas, the error _message_ itself is equivalent.
For collections, the error message for lazy validation is limited and non-deterministic: the error message only includes
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
{class}`~dataframely.Config` context manager) and applies to both eager and lazy validation:

```python
import dataframely as dy

with dy.Config(max_failure_examples=5):
    MySchema.validate(df)
```

For column-level rules, examples include the value in the offending column. For schema-level rules, examples include all
data columns of the schema, except for the `primary_key` rule where examples are limited to the primary key columns.
The default value of `0` disables examples entirely.
