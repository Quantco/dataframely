# Persisting data

`dataframely` provides convenience methods for writing and reading validated data
frames to and from parquet files. These methods dispatch to the polars-native
functionality with little overhead, but adapt the interface to `dataframely`'s
{class}`~dataframely.Collection` and {class}`~dataframely.FailureInfo` types.

```{important}
These methods do **not** persist or inspect any schema metadata and do **not** run
validation when reading. It is the user's responsibility to ensure that the data on
disk is valid. When reading from untrusted sources, call
{meth}`~dataframely.Collection.validate` (or {meth}`~dataframely.Collection.filter`)
explicitly after reading.
```

The following methods are available:

| Class                             | Writing                                                                                       | Reading                                                                                      |
| --------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| {class}`~dataframely.Collection`  | {meth}`~dataframely.Collection.write_parquet`, {meth}`~dataframely.Collection.sink_parquet`   | {meth}`~dataframely.Collection.read_parquet`, {meth}`~dataframely.Collection.scan_parquet`   |
| {class}`~dataframely.FailureInfo` | {meth}`~dataframely.FailureInfo.write_parquet`, {meth}`~dataframely.FailureInfo.sink_parquet` | {meth}`~dataframely.FailureInfo.read_parquet`, {meth}`~dataframely.FailureInfo.scan_parquet` |

## Persisting a {class}`~dataframely.Collection`

A {class}`~dataframely.Collection` groups multiple related data frames. Writing a
collection creates one parquet file per member (named `<member>.parquet`) in the
provided directory. Optional members that are not set are simply skipped.

```python
# Any collection will work
class MyCollection(dy.Collection):
    df1: dy.LazyFrame[MySchema1]
    df2: dy.LazyFrame[MySchema2]


collection = MyCollection.validate(...)

# Writes and reads operate on a directory instead of a single file.
collection.write_parquet("/path/to/directory/")

# Read the members back (no validation is performed).
new_collection = MyCollection.read_parquet("/path/to/directory/")

# ...or lazily
new_collection = MyCollection.scan_parquet("/path/to/directory/")
```

Individual schemas can be persisted directly through the polars-native functions, e.g.
`df.write_parquet(...)` and `pl.read_parquet(...)`. Call
{meth}`~dataframely.Schema.validate` explicitly if you need to (re-)validate the data.

## Persisting a {class}`~dataframely.FailureInfo`

The {class}`~dataframely.FailureInfo` returned by {meth}`~dataframely.Schema.filter`
can be written to and read from a single parquet file. In addition to the invalid rows,
the boolean rule columns are persisted so that methods like
{meth}`~dataframely.FailureInfo.counts` keep working after a round-trip.

```python
_, failure = MySchema.filter(df)

# Write to (or stream into) a parquet file...
failure.write_parquet("failures.parquet")

# ...and read it back.
failure = dy.FailureInfo.read_parquet("failures.parquet")
```
