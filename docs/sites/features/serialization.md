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

A {class}`~dataframely.Schema` controls the contents of a single dataframe. In this case, serialization
means that we store a single dataframe in the storage backend and attach a string representation
of the schema as metadata.

```python
class MySchema(dy.Schema):
    x = dy.Int64(primary_key=True)


df: dy.DataFrame[MySchema] = MySchema.validate(
    pl.DataFrame(
        {"x"[1, 2, 3]}
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

The role of the metadata is that when reading, `dataframely` can internally check
if the `Schema` class we use for reading matches the stored metadata in the file.
If it does, we do not need to run validation again,
but we can infer that the data in the file already matches the schema, which saves us time.

## Serialization in {class}`dy.Collection`

Serialization in collections works analogously to schemas. The only difference is that
we now have to handle multiple dataframes instead of a single one.
`dataframely` will therefore have to create multiple tables in the storage backend
(e.g. multiple parquet files, or multiple delta tables).

```python
# Any collection will work
class MyCollection:
    df1: dy.LazyFrame[MySchema1]
    df2: dy.LazyFrame[MySchema2]


collection: MyCollection = MyCollection.validate(...)

# Writes and reads work the same as for Schema, except that the argument is adapted
# to allow for multiple dataframes,
# e.g. for parquet: Pass a directory instead of a path to a single parquet
collection.write_parquet("/path/to/directory/")
collection.read_parquet("/path/to/directory/")
collection.scan_parquet("/path/to/directory/")
```

Just as for `Schema`, metadata is stored in the backend to encode the schema information.
This includes the schemas of the member dataframes as well as collection-level constraints.

## Configuring validation behavior on reads

All scan / read operations allow the user to specify a `validation` keyword argument
that can be used to define how `dataframely` should react if the schema information
stored in the backend does not match the schema used for reading.
Refer to the API docs linked in the table above for details.
