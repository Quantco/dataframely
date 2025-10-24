# Serialization

`dataframely` provides support for easily storing and reading validated data.
`polars` already provides native support for serializing data frames into different storage
backends. For the storage of the data itself, `dataframely` usually dispatches to polars-native
functionality with little overhead. The distinct feature that `dataframely` offers in addition
to `polars` is that it also stores metadata about the schema of the serialized dataframe. This is useful
because it means that we can avoid having to validate the schema again when reading back a stored data frame.

The `parquet` and `deltalake` backends are currently supported. Wherever possible, lazy and eager operations are
supported.

| Class / Backend support           | parquet                                                                                                                                                                                         | deltalake                                                                                                                                  |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| {class}`~dataframely.Schema`      | {meth}`~dataframely.Schema.write_parquet`, {meth}`~dataframely.Schema.sink_parquet` <br>{meth}`~dataframely.Schema.read_parquet`, {meth}`~dataframely.Schema.scan_parquet`                      | {meth}`~dataframely.Schema.write_delta` <br> {meth}`~dataframely.Schema.read_delta`, {meth}`~dataframely.Schema.scan_delta`                |
| {class}`~dataframely.Collection`  | {meth}`~dataframely.Collection.write_parquet`, {meth}`~dataframely.Collection.sink_parquet` <br> {meth}`~dataframely.Collection.read_parquet`, {meth}`~dataframely.Collection.scan_parquet`     | {meth}`~dataframely.Collection.write_delta` <br> {meth}`~dataframely.Collection.read_delta`, {meth}`~dataframely.Collection.scan_delta`    |
| {class}`~dataframely.FailureInfo` | {meth}`~dataframely.FailureInfo.write_parquet`, {meth}`~dataframely.FailureInfo.sink_parquet` <br> {meth}`~dataframely.FailureInfo.read_parquet`, {meth}`~dataframely.FailureInfo.scan_parquet` | {meth}`~dataframely.FailureInfo.write_delta` <br> {meth}`~dataframely.FailureInfo.read_delta`, {meth}`~dataframely.FailureInfo.scan_delta` |

## Serialization in {class}`~dataframely.Schema`

A {class}`~dataframely.Schema` controls the contents of a single data frame. In this case, serialization
means that we store a single data frame in the storage backend and attach a string representation
of the schema as metadata.

```python
class MySchema(dy.Schema):
    x = dy.Int64(primary_key=True)


df: dy.DataFrame[MySchema] = MySchema.validate(
    pl.DataFrame(
        {"x": [1, 2, 3]}
    )
)

# The serialization methods provide interfaces that are as close as possible to the
# polars interface you are probably familiar with
# Writing to parquet
MySchema.write_parquet(df, "my.parquet")

# Or to deltalake
MySchema.write_delta(df, "/path/to/table")
```

Then, we can read back the data:

```python
# Reading parquet eagerly
new_df: dy.DataFrame[MySchema] = MySchema.read_parquet("my.parquet")

# ...or lazily
new_lf: dy.LazyFrame[MySchema] = MySchema.scan_parquet("my.parquet")

# Or deltalake eagerly
new_df: dy.DataFrame[MySchema] = MySchema.read_delta("/path/to/table")

# ...or lazily
new_lf: dy.LazyFrame[MySchema] = MySchema.scan_delta("/path/to/table")
```

Using the stored metadata, `dataframely` can internally check
if the `Schema` class we use for reading matches the stored metadata in the file.
If it does, we do not need to run validation again,
but we can infer that the data in the file already matches the schema, which saves us time.

## Serialization in {class}`~dataframely.Collection`

Serialization in collections works analogously to schemas. The only difference is that
we now have to handle multiple data frames instead of a single one.
`dataframely` will therefore create multiple tables in the storage backend
(e.g. multiple parquet files, or multiple delta tables).

```python
# Any collection will work
class MyCollection:
    df1: dy.LazyFrame[MySchema1]
    df2: dy.LazyFrame[MySchema2]


collection: MyCollection = MyCollection.validate(...)

# Writes and reads work the same as for Schema, except that the argument is adapted
# to allow for multiple data frames,
# e.g. for parquet: Pass a directory instead of a path to a single parquet
collection.write_parquet("/path/to/directory/")
collection.read_parquet("/path/to/directory/")
collection.scan_parquet("/path/to/directory/")
```

Just as for `Schema`, metadata is stored in the backend to encode the schema information.
This includes the schemas of the member data frames as well as collection-level constraints.

## What happens if the stored metadata is missing or wrong?

All scan / read operations allow the user to specify a `validation` keyword argument
that can be used to define how `dataframely` should react if there is no schema information
found in the backend, or if the schema information does not match the the schema used for reading.
By default, `dataframely` will run validation and emit a warning in this case.
If you instead

Refer to the API docs linked in the table above for details. If you want to avoid the warning,
pass `validation="allow"`, e.g.:

```python
# Will not warn and only validate if necessary
MySchema.read_parquet("my.parquet", validation="allow")

# Will raise an error if validation cannot be skipped
MySchema.read_parquet("my.parquet", validation="forbid")

# Dangerous: Will never validate. It's possible to load data that violates the schema!
MySchema.read_parquet("my.parquet", validation="forbid")
```

```{note}
Some schema information such as data types is trivial to serialize.
However, we also serialize custom schema rules.
For this, we rely on `polars.Expression.meta.serialize`, which is not currently guaranteed
to be stable between polars version. As a result, it is possible that `polars`
version updates can break our ability to recognize a stored schema, even if it still
semantically matches the current schema. This situation is treated the same
as if no stored schema was found.
```

## Under the hood: what does the metadata look like?

Both `Schema`s and `Collection`s implement public methods `serialize` that return a string-serialized version of the
metadata for this object. In addition to the internal usage of these methods in `dataframely`, they can also be useful
if you want an easily parseable representation of your schema,
for example to generate schema diffs.

Here's an example schema:

```python
class HouseSchema(dy.Schema):
    zip_code = dy.String(nullable=False, min_length=3)
    num_bedrooms = dy.UInt8(nullable=False)
    num_bathrooms = dy.UInt8(nullable=False)
    price = dy.Float64(nullable=False)

    @dy.rule()
    def reasonable_bathroom_to_bedrooom_ratio(cls) -> pl.Expr:
        ratio = pl.col("num_bathrooms") / pl.col("num_bedrooms")
        return (ratio >= 1 / 3) & (ratio <= 3)


```

Calling {meth}`~dataframely.Schema.serialize` returns a string-encoded JSON representation of the schema:

```python
json.loads(HouseSchema.serialize())

{'columns': {'num_bathrooms': {'check': None,
                               'column_type': 'UInt8',
                               'is_in': None,
                               'max': None,
                               'max_exclusive': None,
                               'metadata': None,
                               'min': None,
                               'min_exclusive': None,
                               'nullable': False,
                               'primary_key': False},
             'num_bedrooms': {'check': None,
                              'column_type': 'UInt8',
                              'is_in': None,
                              'max': None,
                              'max_exclusive': None,
                              'metadata': None,
                              'min': None,
                              'min_exclusive': None,
                              'nullable': False,
                              'primary_key': False},
             'price': {'allow_inf_nan': False,
                       'check': None,
                       'column_type': 'Float64',
                       'max': None,
                       'max_exclusive': None,
                       'metadata': None,
                       'min': None,
                       'min_exclusive': None,
                       'nullable': False,
                       'primary_key': False},
             'zip_code': {'check': None,
                          'column_type': 'String',
                          'max_length': None,
                          'metadata': None,
                          'min_length': 3,
                          'nullable': False,
                          'primary_key': False,
                          'regex': None}},
 'name': 'HouseSchema',
 'rules': {'reasonable_bathroom_to_bedrooom_ratio': {'expr': {'__type__': 'expression',
                                                              'value': 'gapCaW5hcnlFeHByg6RsZWZ0gapCaW5hcnlFeHByg6RsZWZ0gapCaW5hcnlFeHByg6RsZWZ0gaZDb2x1bW6tbnVtX2JhdGhyb29tc6JvcKpUcnVlRGl2aWRlpXJpZ2h0gaZDb2x1bW6sbnVtX2JlZHJvb21zom9wpEd0RXGlcmlnaHSBp0xpdGVyYWyBo0R5boGlRmxvYXTLP9VVVVVVVVWib3CjQW5kpXJpZ2h0gapCaW5hcnlFeHByg6RsZWZ0gapCaW5hcnlFeHByg6RsZWZ0gaZDb2x1bW6tbnVtX2JhdGhyb29tc6JvcKpUcnVlRGl2aWRlpXJpZ2h0gaZDb2x1bW6sbnVtX2JlZHJvb21zom9wpEx0RXGlcmlnaHSBp0xpdGVyYWyBo0R5boGjSW50xBAAAAAAAAAAAAAAAAAAAAAD'},
                                                     'rule_type': 'Rule'}},
 'versions': {'dataframely': '2.0.0', 'format': '1', 'polars': '1.33.1'}}
```

Note that while most of the serialized schema is straightforward, the encoding of rules requires serialization of polars
expression.
See {class}`~dataframely._serialization.SchemaJSONEncoder` for implementation details.

This functionality works equivalently for {meth}`~dataframely.Collection.serialize`.
