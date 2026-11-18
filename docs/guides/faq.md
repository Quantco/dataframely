# FAQ

Whenever you find out something that you were surprised by or needed some non-trivial
thinking, please add it here.

## How do I define additional unique keys in a {class}`~dataframely.Schema`?

`dataframely` only supports defining a single non-nullable (composite) primary key in {class}`~dataframely.Schema`.
However, in some scenarios it may be useful to define additional unique keys.

For individual columns, set `unique=True` to require that the column's values are unique. Note that multiple `null`
values are considered duplicates. To require uniqueness only for non-null values, or uniqueness across a combination of
columns, use a custom rule instead:

```python
class UserSchema(dy.Schema):
    user_id = dy.UInt64(primary_key=True)
    username = dy.String(unique=True)
    email = dy.String(nullable=True)  # Must be unique, or null.

    @dy.rule()
    def unique_email_or_null(cls) -> pl.Expr:
        """Email must be unique, if provided."""
        return pl.col("email").is_null() | pl.col("email").is_unique()
```

## How do I fix the ruff error `First argument of a method should be named self`?

See our documentation on [custom rules](./quickstart.md#custom-rules).

## How do I get a {class}`~dataframely.Schema` for my dataframe?

You can use {func}`dataframely.experimental.infer_schema` to get a basic {class}`~dataframely.Schema` definition for
your dataframe. The function infers column names, types and nullability from the dataframe and returns a code
representation of a {class}`~dataframely.Schema`
Starting from this definition, you can then refine the schema by adding additional rules, adjusting types, etc.

## What versions of `polars` does `dataframely` support?

Our CI automatically tests `dataframely` for a minimal supported version of `polars`, which is currently `1.35.*`,
and the latest stable version.
Python 3.15 and later require `polars>=2` for Python 3.15 compatibility.
We aim to extend support for new `polars` versions as they are released.

If your `polars` version is not in the range of supported versions, `dataframely` may still work, but you may
encounter unexpected issues.
