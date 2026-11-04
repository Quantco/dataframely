# Extra columns

By default, {meth}`~dataframely.Schema.validate`, {meth}`~dataframely.Schema.filter` and
{meth}`~dataframely.Schema.cast` drop every column that the schema does not define. When a data frame has
many columns and only some of them need to be validated (e.g. a wide feature table), this forces you to either
declare all columns or re-attach the remaining ones after validation.

The `extra_columns` setting controls how columns that are not defined in the schema are treated:

- `"drop"` (default): Remove them from the output.
- `"raise"`: Raise a {class}`~dataframely.SchemaError` that lists them.
- `"keep"`: Keep them in the output, after the schema's columns and in their input order. Kept columns are not
  validated.

## Configuring a schema

Set the default for a schema with the `extra_columns` class keyword. Subclasses inherit it and may override it:

```python
class FeatureSchema(dy.Schema, extra_columns="keep"):
    claim_id = dy.String(primary_key=True)
    cost = dy.Int64()


df = pl.DataFrame({"claim_id": ["a"], "cost": [1], "feature_1": [0.5]})
FeatureSchema.validate(df).columns  # ["claim_id", "cost", "feature_1"]
```

## Overriding the setting for a single call

All of {meth}`~dataframely.Schema.validate`, {meth}`~dataframely.Schema.is_valid`,
{meth}`~dataframely.Schema.filter` and {meth}`~dataframely.Schema.cast` accept `extra_columns` to override the
schema's setting:

```python
FeatureSchema.validate(df, extra_columns="drop").columns  # ["claim_id", "cost"]
FeatureSchema.validate(df, extra_columns="raise")  # raises SchemaError
```

```{note}
Kept columns are carried along during validation. An extra column whose name clashes with a name used internally
for validation (such as the name of a rule) cannot be kept and raises a {class}`~dataframely.SchemaError`.
```
