# Column Metadata

Sometimes it can be useful to attach user-provided metadata to columns of tables.
The `metadata` parameter is available for all column types and accepts a dictionary of arbitrary objects.
For instance, one may use the `metadata` parameter to mark a column as pseudonymized or provide other context-specific information.

```python
class UserSchema(dy.Schema):
    id = dy.String(primary_key=True)
    # Mark last name column as pseudonymized and (non-docstring) comment on it.
    last_name = dy.String(metadata={
        "pseudonymized": True,
        "comment": "Pseudonymized using cryptographic hash function"
    })
    # Add information about database column type.
    address = dy.String(metadata={"database-type": "VARCHAR(MAX)"})
```

```python
>>> print(UserSchema.last_name.metadata)
{'pseudonymized': True, 'comment': 'Pseudonymized using cryptographic hash function'}
```

Metadata are never read by `dataframely` and merely enable users to provide custom information
in a structured way.

```{note}
Experience has shown that user-provided metadata can be useful for code generation, where, for instance, SQL code is generated from `dataframely` schemas.
```
