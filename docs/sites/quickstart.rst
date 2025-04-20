Quickstart
==========

For the purpose of this guide, let's assume that we're working with data that we use to predict housing prices.
To this end, we want to ensure that all the data we're using meets several expectations.
As a running example, consider the following data set:

.. list-table::
    :header-rows: 1

    * - ``zip_code``
      - ``num_bedrooms``
      - ``num_bathrooms``
      - ``price``
    * - "01234"
      - 2
      - 1
      - 100,000
    * - "01234"
      - 2
      - 2
      - 110,000
    * - "1"
      - 1
      - 1
      - 50,000
    * - "213"
      - <NULL>
      - 1
      - 80,000
    * - "123"
      - <NULL>
      - 0
      - 60,000
    * - "213"
      - 2
      - 8
      - 160,000


Creating a :class:`~dataframely.Schema` class
---------------------------------------------------

To get started with dataframely, you'll always want to define a schema (``dataframely.Schema``). For example, we might set up the following:

::

    import dataframely as dy

    class HouseSchema(dy.Schema):
        zip_code = dy.String(nullable=False, min_length=3)
        num_bedrooms = dy.UInt8(nullable=False)
        num_bathrooms = dy.UInt8(nullable=False)
        price = dy.Float64(nullable=False)

This translates into the following expectations on our data:

- We require exactly four columns ``zip_code``, ``num_bedrooms``, ``num_bathrooms``, ``price``
- We expect a particular data type for each of these, requiring all of them to be non-nullable
- The zip code must be at least three characters as we consider any other zip code invalid

Custom rules
------------

While parameters in the column-initializers allow for defining expectations on a single column (e.g. ``min_length`` for string columns), this might not always by sufficient.
In many cases, we want to check expectations across columns: for example, the ratio between the number of bathrooms and bedrooms should not be too high.

In ``dataframely``, we can do this by adding a custom rule to our schema:

::

    import dataframely as dy

    class HouseSchema(dy.Schema):
        zip_code = dy.String(nullable=False, min_length=3)
        num_bedrooms = dy.UInt8(nullable=False)
        num_bathrooms = dy.UInt8(nullable=False)
        price = dy.Float64(nullable=False)

        @dy.rule()
        def reasonable_bathroom_to_bedrooom_ratio() -> pl.Expr:
            ratio = pl.col("num_bathrooms") / pl.col("num_bedrooms")
            return (ratio >= 1 / 3) & (ratio <= 3)

The decorator ``@dy.rule()`` "registers" the function as a rule using its name (i.e. ``reasonable_bathroom_to_bedrooom_ratio``).
The returned expression provides a boolean value for each row of the data which evaluates to ``True`` whenever the data are valid with respect to this rule.

Group rules
-----------

For defining even more complex rules, the ``@dy.rule`` decorator allows for a ``group_by``
parameter: this allows to evaluate a rule across *rows*.
For our housing data, this allows us to specify, for example, that we want to observe at least two houses per zip code:

::

    import dataframely as dy

    class HouseSchema(dy.Schema):
        zip_code = dy.String(nullable=False, min_length=3)
        num_bedrooms = dy.UInt8(nullable=False)
        num_bathrooms = dy.UInt8(nullable=False)
        price = dy.Float64(nullable=False)

        @dy.rule()
        def reasonable_bathroom_to_bedrooom_ratio() -> pl.Expr:
            ratio = pl.col("num_bathrooms") / pl.col("num_bedrooms")
            return (ratio >= 1 / 3) & (ratio <= 3)

        @dy.rule(group_by=["zip_code"])
        def minimum_zip_code_count() -> pl.Expr:
            return pl.len() >= 2

When defining rules on groups, we have to take care to use some kind of "aggregate function"
in order to produce exactly one value per group:
in group rules, the "input" that the expression is evaluated on is a set of rows.

.. note::
    If you are using `mypy` to type check your code, the usage of rules requires you to use the dataframely `mypy` plugin. In order to use it, add the following to your ``pyproject.toml``:

    ::

        [tool.mypy]
        plugins = ["dataframely.mypy"]


Validating data against a schema
--------------------------------

Once we're done defining our schema, we want to apply the schema to our data.
To validate data against a schema, we can use the ``validate`` method of the schema class. For example, we can validate the data set above as follows:

::

    import polars as pl

    df = pl.DataFrame({
        "zip_code": ["01234", "01234", "1", "213", "123", "213"],
        "num_bedrooms": [2, 2, 1, None, None, 2],
        "num_bathrooms": [1, 2, 1, 1, 0, 8],
        "price": [100_000, 110_000, 50_000, 80_000, 60_000, 160_000]
    })

    # Validate the data and cast columns to expected types
    validated_df = HouseSchema.validate(df, cast=True)

If any row in ``df`` is invalid, i.e., any rule defined on individual columns or the entire schema evaluates to ``False``, a validation exception is raised.
Here, we have invalid data in the ``num_bedrooms`` and ``zip_code`` columns.

::

    RuleValidationError: 2 rules failed validation:
    * Column 'num_bedrooms' failed validation for 1 rules:
    - 'nullability' failed for 2 rows
    * Column 'zip_code' failed validation for 1 rules:
    - 'min_length' failed for 1 rows

Otherwise, if all rows in ``df`` are valid, ``validate`` returns a validated data frame of type ``dy.DataFrame[HouseSchema]``.
The same applies when a ``pl.LazyFrame`` is passed to `validate`.
The generic data frame types allow for more readable function signatures to express
expectations on the schema of the data frame, e.g.:

::

    def train_model(df: dy.DataFrame[HouseSchema]) -> None:
        ...

The type checker (typically ``mypy``) then ensures that it is actually a
``dy.DataFrame[HouseSchema]`` that is passed to the function and it complains if a plain
(i.e., non-validated) ``pl.DataFrame`` or a data frame with a different schema is used.
The ``train_model`` function can be implemented with peace of mind: ``df`` looks exactly as needed.

.. note::
    Make sure that you do not bypass the type checker by using ``# type: ignore`` annotations in these contexts.
    This defies the entire purpose of the typed data frames.
    Also note that the frame types generic over a schema are *only* available to the static type checker.
    If you call ``isinstance()`` checking for ``dy.DataFrame``, it will *always* evaluate to ``False``.
    The run-time type of the data frame is still a ``pl.DataFrame``.

Using soft-validation to introspect validation failures
-------------------------------------------------------

While ``validate`` is useful for ensuring that the entire dataset meets expectations,
it is not always useful in production systems where invalid rows should be ignored while all valid rows should be salvaged.

To this end, ``dataframely`` provides the ``filter`` method that performs "soft-validation":

::

    # Filter the data and cast columns to expected types
    good, failure = HouseSchema.filter(df, cast=True)

    # Inspect the reasons for the failed rows
    print(failure.counts())

In this case, ``good`` remains to be a ``dy.DataFrame[HouseSchema]``, albeit with potentially fewer rows than ``df``.
The ``failure`` object is of type :class:`~dataframely.FailureInfo` and provides means to inspect
the reasons for validation failures for invalid rows.

Given the example data above and the schema that we defined, we know that rows 2, 3, 4, and 5 are invalid (0-indexed):

- Row 2 has a zip code that does not appear at least twice
- Row 3 has a NULL value for the number of bedrooms
- Row 4 violates both of the rules above
- Row 5 violates the reasonable bathroom to bedroom ratio

Using the ``counts`` method on the :class:`~dataframely.FailureInfo` object will result in the following dictionary:

::

    {
      "reasonable_bathroom_to_bedrooom_ratio": 1,
      "minimum_zip_code_count": 2,
      "zip_code|min_length": 1,
      "num_bedrooms|nullability": 2,
    }


To get a data frame containing all failed rows, we can use the ``invalid`` method:

::

    failed_df = failure.invalid()

This information tends to be very useful in tracking down issues with the data,
both in productive systems and analytics environments.

Type casting
------------

In rare cases, you might already be *absolutely certain* that a data frame is valid with
respect to a particular schema and do not want to pay the runtime cost of calling ``validate`` or ``filter``.
To this end, you can use the ``cast`` method to tell this to the type checker without inspecting the contents of the data frame:

::

    df_valid = HouseSchema.cast(df)

A use case for ``cast`` could be the concatenation of two data frames with known schema, e.g.:

::

    df1: dy.DataFrame[HouseSchema]
    df2: dy.DataFrame[HouseSchema]
    df_concat = HouseSchema.cast(pl.concat([df1, df2]))


Integration with external tools
-------------------------------

Lastly, ``dataframely`` schemas can be used to integrate with external tools:

- ``HouseSchema.create_empty()`` creates an empty ``dy.DataFrame[HouseSchema]`` that can be used for testing
- ``HouseSchema.sql_schema()`` provides a list of `sqlalchemy <https://www.sqlalchemy.org>`_ columns that can be used to create SQL tables using types and constraints in line with the schema
- ``HouseSchema.pyarrow_schema()`` provides a `pyarrow <https://arrow.apache.org/docs/python/index.html>`_ schema with appropriate column dtypes and nullability information


Outlook
-------

This concludes the quickstart guide. For more information, please see the `real-world example <examples/real-world.ipynb>`__ or dive into the API documentation.
