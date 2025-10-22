# Generating SQL schema definitions

It is often useful to store data in a SQL-based database server. `dataframely` aims to make this easy by
providing a simple mechanism for translating your `dataframely` schemas to SQL table definitions.

There are many different flavors of SQL syntax. To avoid reinventing the wheel, we use [
`sqlalchemy`](https://www.sqlalchemy.org/) as an abstraction
layer between python and SQL.

## Individual tables

The main functionality `dataframely` offers is that it converts your `dy.Schema` to a collection of `sqlalchemy.Column`:

```python
import dataframely as dy
import sqlalchemy as sa


class MySchema(dy.Schema):
    x = dy.Int64(primary_key=True)
    y = dy.String(nullable=False)


engine = sa.create_engine(...)
columns: list[sa.Column] = MySchema.to_sqlalchemy_columns(engine.dialect)
```

You can then do with the columns what you please. Most likely, you want to create a table with them:

```python
my_table = sa.Table("myTable", sa.MetaData(), *columns)
my_table.create(engine)
```

You can also inspect the SQL code that `sqlalchemy` would execute:

```python
from sqlalchemy.schema import CreateTable

print(CreateTable(my_table).compile())
```

In the example case, this renders to:

```SQL
CREATE TABLE "myTable"
(
    x BIGINT  NOT NULL,
    y VARCHAR NOT NULL,
    PRIMARY KEY (x)
)
```

Uploading data can then be handled by {meth}`polars.DataFrame.write_database`:

```python
df: dy.DataFrame[MySchema]

df.write_database(
    connection=engine,
    table_name=my_table.name,
    if_table_exists="append"
)
```

```{note}
**Why do you need to pass in the SQL dialect?** Even though `sqlalchemy` handles most dialect dependencies, we sometimes still need to intervene. For example, when using Microsoft SQL Server, `sqlalchemy` will render the `sqlalchemy.Date` type into a raw SQL `DATETIME`, while we think that `DATE` would be more appropriate.
```

```{note}
**Implementation:** The choice of `sqlalchemy` type is implemented in {meth}`~dataframely.Column.sqlalchemy_dtype`, which is overwritten by each of the subtypes of {class}`~dataframely.Column`. For example, the implementation for {class}`~dataframely.Date` is {meth}`~dataframely.Date.sqlalchemy_dtype`.
```

```{note}
**Constraints:** The nullability and primary key constraints you define in `dataframely` are translated to SQL. Custom filters and rules are not.
```

```{note}
**Length of string columns:** For string columns, `dataframely` will attempt to pass information about the maximal length into the SQL definition. This is trivial if `max_length` is set. Otherwise, if a `regex` is provided,
the maximal length of the string is inferred from the regular expression if possible. Note that having inferable
maximal lengths can be particularly important for primary key columns. Some database systems, such as Microsoft SQL Server, do not allow `VARCHAR(max)` columns (unbounded strings) to be used as primary keys.
```

## Collections of multiple tables

If you have an entire `dy.Collection`, it's also easy to generate one table for each member table of the collection.
`sqlalchemy.MetaData` is a commonly used container in such scenarios:

```python
MyCollection: dy.Collection
meta = sa.MetaData()
for name, dy_schema in MyCollection.member_schemas().items():
    sa.Table(
        name,
        meta,
        *dy_schema.to_sqlalchemy_columns(dialect=engine.dialect),
    )
meta.create_all()
```
