# FAQ

Whenever you find out something that you were surprised by or needed some non-trivial
thinking, please add it here.

## How do I define additional unique keys in a {class}`~dataframely.Schema`?

By default, `dataframely` only supports defining a single non-nullable (composite) primary key in :class:
`~dataframely.Schema`.
However, in some scenarios it may be useful to define additional unique keys (which support nullable fields and/or which
are additionally unique).

Consider the following example, which demonstrates two rules: one for validating that a field is entirely unique, and
another for validating that a field, when provided, is unique.

```python
class UserSchema(dy.Schema):
    user_id = dy.UInt64(primary_key=True, nullable=False)
    username = dy.String(nullable=False)
    email = dy.String(nullable=True)  # Must be unique, or null.

    @dy.rule(group_by=["username"])
    def unique_username(cls) -> pl.Expr:
        """Username, a non-nullable field, must be total unique."""
        return pl.len() == 1

    @dy.rule()
    def unique_email_or_null(cls) -> pl.Expr:
        """Email must be unique, if provided."""
        return pl.col("email").is_null() | pl.col("email").is_unique()
```

## How do I fix the ruff error `First argument of a method should be named self`?

If you are using [`ruff`](https://docs.astral.sh/ruff/) and introduce custom rules for your schemas, `ruff` will create
the following linting error:

```
N805 First argument of a method should be named `self`
```

To fix this, you'll need to let `ruff` know that the `@dy.rule` decorator is applied to classmethods. This can easily
be done by adding the following to your `pyproject.toml`:

```toml
[tool.ruff.lint.pep8-naming]
classmethod-decorators = ["dataframely.rule"]
```
