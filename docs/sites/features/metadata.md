# Metadata

Sometimes it can be useful to attach user-provided metadata to columns of tables.
The `metadata` parameter is available for all column types and takes a dictionary of arbitrary objects.
For instance, one may use the `metadata` parameter to mark a column as pseudonymized or provide other context-specific information.

```python
class UserSchema(dy.Schema):
    id = dy.String(primary_key=True)
    # Mark last name column as pseudonymized.
    last_name = dy.String(metadata={"pseudonymized": True})
    # Add information about database column type.
    address = dy.String(metadata={"database-type": "VARCHAR(MAX)"})
```

Metadata are never read by `dataframely` and merely enable users to provide custom information
in a structured way.

```{note}
Experience has shown that user-provided metadata can be useful for code generation, where, for instance, SQL code is generated from `dataframely` schemas.
```
