# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import warnings
from abc import ABC
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Literal, Self, overload

import polars as pl
import polars.exceptions as plexc
import polars.selectors as cs
from polars._typing import FileSource, PartitioningScheme

from ._base_schema import BaseSchema
from ._compat import pa, sa
from ._rule import Rule, rule_from_dict, with_evaluation_rules
from ._serialization import (
    SCHEMA_METADATA_KEY,
    SERIALIZATION_FORMAT_VERSION,
    SchemaJSONDecoder,
    SchemaJSONEncoder,
    serialization_versions,
)
from ._typing import DataFrame, LazyFrame, Validation
from ._validation import DtypeCasting, validate_columns, validate_dtypes
from .columns import Column, column_from_dict
from .config import Config
from .exc import RuleValidationError, ValidationError, ValidationRequiredError
from .failure import FailureInfo
from .random import Generator

_ORIGINAL_NULL_SUFFIX = "__orig_null__"


# ------------------------------------------------------------------------------------ #
#                                   SCHEMA DEFINITION                                  #
# ------------------------------------------------------------------------------------ #


class Schema(BaseSchema, ABC):
    """Base class for all custom data frame schema definitions.

    A custom schema should only define its columns via simple assignment:

    .. code-block:: python

        class MySchema(Schema):
            a = dataframely.Int64()
            b = dataframely.String()

    All definitions using non-datatype classes are ignored.

    Schemas can also be nested (arbitrarily deeply): in this case, the columns defined
    in the subclass are simply appended to the columns in the superclass(es).
    """

    # ----------------------------------- CREATION ----------------------------------- #

    @classmethod
    @overload
    def create_empty(cls, *, lazy: Literal[False] = False) -> DataFrame[Self]: ...

    @classmethod
    @overload
    def create_empty(cls, *, lazy: Literal[True]) -> LazyFrame[Self]: ...

    @classmethod
    @overload
    def create_empty(cls, *, lazy: bool) -> DataFrame[Self] | LazyFrame[Self]: ...

    @classmethod
    def create_empty(cls, *, lazy: bool = False) -> DataFrame[Self] | LazyFrame[Self]:
        """Create an empty data or lazy frame from this schema.

        Args:
            lazy: Whether to create a lazy data frame. If ``True``, returns a lazy frame
                with this Schema. Otherwise, returns an eager frame.

        Returns:
            An instance of :class:`polars.DataFrame` or :class:`polars.LazyFrame` with
            this schema's defined columns and their data types.
        """
        df = pl.DataFrame(
            schema={name: col.dtype for name, col in cls.columns().items()},
        )
        if lazy:
            return cls.cast(df.lazy())
        return cls.cast(df)

    @classmethod
    @overload
    def create_empty_if_none(
        cls,
        df: DataFrame[Self] | None,
        *,
        lazy: Literal[False] = False,
    ) -> DataFrame[Self]: ...

    @classmethod
    @overload
    def create_empty_if_none(
        cls,
        df: LazyFrame[Self] | None,
        *,
        lazy: Literal[True],
    ) -> LazyFrame[Self]: ...

    @classmethod
    @overload
    def create_empty_if_none(
        cls,
        df: DataFrame[Self] | LazyFrame[Self] | None,
        *,
        lazy: bool,
    ) -> DataFrame[Self] | LazyFrame[Self]: ...

    @classmethod
    def create_empty_if_none(
        cls,
        df: DataFrame[Self] | LazyFrame[Self] | None,
        *,
        lazy: bool = False,
    ) -> DataFrame[Self] | LazyFrame[Self]:
        """Impute ``None`` input with an empty, schema-compliant lazy or eager data
        frame or return the input as lazy or eager frame.

        Args:
            df: The data frame to check for ``None``. If it is not ``None``, it is
                returned as lazy or eager frame. Otherwise, a schema-compliant data
                or lazy frame with no rows is returned.
            lazy: Whether to return a lazy data frame. If ``True``, returns a lazy frame
                with this Schema. Otherwise, returns an eager frame.

        Returns:
            The given data frame ``df`` as lazy or eager frame, if it is not ``None``.
            An instance of :class:`polars.DataFrame` or :class:`polars.LazyFrame` with
            this schema's defined columns and their data types, but no rows, otherwise.
        """
        if df is not None:
            return df.lazy() if lazy else df.lazy().collect()

        return cls.create_empty(lazy=lazy)

    # ----------------------------------- SAMPLING ----------------------------------- #

    @classmethod
    def sample(
        cls,
        num_rows: int | None = None,
        *,
        overrides: (
            Mapping[str, Iterable[Any]] | Sequence[Mapping[str, Any]] | None
        ) = None,
        generator: Generator | None = None,
    ) -> DataFrame[Self]:
        """Create a random data frame with a predefined number of rows.

        Generally, **this method should only be used for testing**. Also, if you want
        to generate _realistic_ test data, it is inevitable to implement your custom
        sampling logic (by making use of the :class:`Generator` class).

        In order to allow for sampling random data frames in the presence of custom
        rules and primary key constraints, this method performs *fuzzy sampling*: it
        samples in a loop until it finds a data frame of length ``num_rows`` which
        adhere to the schema. The maximum number of sampling rounds is configured via
        ``max_sampling_iterations`` in the :class:`Config` class. By fixing this setting
        to 1, it is only possible to reliably sample from schemas without custom rules
        and without primary key constraints.

        Args:
            num_rows: The (optional) number of rows to sample for creating the random
                data frame. Must be provided (only) if no ``overrides`` are provided. If
                this is ``None``, the number of rows in the data frame is determined by
                the length of the values in ``overrides``.
            overrides: Fixed values for a subset of the columns of the sampled data
                frame. Just like when initializing a :mod:`polars.DataFrame`, overrides
                may either be provided as "column-" or "row-layout", i.e. via a mapping
                or a list of mappings, respectively. The number of rows in the result
                data frame is equal to the length of the values in ``overrides``. If both
                ``overrides`` and ``num_rows`` are provided, the length of the values in
                ``overrides`` must be equal to ``num_rows``. The order of the items is
                guaranteed to match the ordering in the returned data frame. When providing
                values for a column, no sampling is performed for that column.
            generator: The (seeded) generator to use for sampling data. If ``None``, a
                generator with random seed is automatically created.

        Returns:
            A data frame valid under the current schema with a number of rows that matches
            the length of the values in ``overrides`` or ``num_rows``.

        Raises:
            ValueError: If ``num_rows`` is not equal to the length of the values in
                ``overrides``.
            ValueError: If no valid data frame can be found in the configured maximum
                number of iterations.

        Attention:
            Be aware that, due to sampling in a loop, the runtime of this method can be
            significant for complex schemas. Consider passing a seeded generator and
            evaluate whether the runtime impact in the tests is bearable. Alternatively,
            it can be beneficial to provide custom column overrides for columns
            associated with complex validation rules.
        """
        g = generator or Generator()

        # Precondition: valid overrides. We put them into a data frame to remember which
        # values have been used in the algorithm below.
        if overrides:
            override_keys = (
                set(overrides) if isinstance(overrides, Mapping) else set(overrides[0])
            )
            column_names = set(cls.column_names())
            if not override_keys.issubset(column_names):
                raise ValueError(
                    f"Values are provided for columns {override_keys - column_names} "
                    "which are not in the schema."
                )

            values = pl.DataFrame(
                overrides,
                schema={
                    name: col.dtype
                    for name, col in cls.columns().items()
                    if name in override_keys
                },
                orient="col" if isinstance(overrides, Mapping) else "row",
            )
            if num_rows is not None and num_rows != values.height:
                raise ValueError(
                    "`num_rows` is different from the length of the provided overrides."
                )
            num_rows = values.height
        else:
            # In case that neither `num_rows` nor `overrides` are provided, fall back to `1`
            if num_rows is None:
                num_rows = 1

            # NOTE: Code becomes rather ugly when allowing `values` to be `None`. Hence,
            #  we're using an empty data frame here and branch on the height of the data
            #  frame.
            values = pl.DataFrame()

        # Prepare expressions for columns that need to be preprocessed during sampling
        # iterations.
        sampling_overrides = cls._sampling_overrides()
        if superfluous_overrides := sampling_overrides.keys() - cls.columns():
            raise ValueError(
                "The schema defines `_sampling_overrides` for columns that are not in the "
                f"schema: {superfluous_overrides}."
            )

        override_expressions = [
            # Cast needed as column pre-processing might change the data types of a column
            expr.cast(cls.columns()[col].dtype).alias(col)
            for col, expr in sampling_overrides.items()
            # Only pre-process columns that are in the schema and were not provided
            # through `overrides`.
            if col not in values.columns
        ]

        # During sampling, we need to potentially sample many times if the schema has
        # (complex) rules.
        #
        # At the same time, we need to ensure that the overrides provided by the user
        # keep their order. To this end, we add a row index to the `values` data frame
        # and re-order the result accordingly.
        #
        # NOTE: One option to potentially run fewer loops would be to sample more than
        #  `n` elements. However, we cannot simply slice the result as that could
        #  potentially violate group rules. Hence, we're bound to calling `filter` on
        #  dataframes with length `n`.
        values = values.with_row_index(name="__row_index__")

        result, used_values, remaining_values = cls._sample_filter(
            num_rows,
            g,
            previous_result=cls.create_empty(),
            used_values=values.slice(0, 0),
            remaining_values=values,
            override_expressions=override_expressions,
        )

        sampling_rounds = 1
        while len(result) != num_rows:
            if sampling_rounds >= Config.options["max_sampling_iterations"]:
                raise ValueError(
                    f"Sampling exceeded {Config.options['max_sampling_iterations']} "
                    "iterations. Consider increasing the maximum number of sampling "
                    "iterations via `dy.Config` or implement your custom sampling "
                    "logic. Alternatively, passing predefined value to `overrides` "
                    "or implementing `_sampling_overrides` for your schema can also "
                    "help the sampling procedure find a valid data frame."
                )
            result, used_values, remaining_values = cls._sample_filter(
                num_rows - len(result),
                g,
                previous_result=result,
                used_values=used_values,
                remaining_values=remaining_values,
                override_expressions=override_expressions,
            )
            sampling_rounds += 1

        if len(used_values) > 0:
            # If we used any values, we want to re-order the result to adhere to the
            # input ordering
            result = (
                pl.concat(
                    [result, used_values.select("__row_index__")], how="horizontal"
                )
                .sort(by="__row_index__")
                .drop("__row_index__")
            )

        # NOTE: There's no need for an additional `validate` or `cast` here since this
        #  is just a re-ordered version of a data frame that was returned from `filter`.
        #  Row order does not affect the validity of a data frame.
        return result  # type: ignore

    @classmethod
    def _sample_filter(
        cls,
        num_rows: int,
        generator: Generator,
        previous_result: pl.DataFrame,
        used_values: pl.DataFrame,
        remaining_values: pl.DataFrame,
        override_expressions: list[pl.Expr],
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Private method to sample a data frame with the schema including subsequent
        filtering.

        Returns:
            The filtered data frame, the used values, and the remaining values.
        """
        sampled = pl.DataFrame(
            {
                name: (
                    col.sample(generator, num_rows)
                    if name not in remaining_values.columns
                    else remaining_values.get_column(name)
                )
                for name, col in cls.columns().items()
            }
        )

        combined_dataframe = pl.concat([previous_result, sampled])
        # Pre-process columns before filtering.
        combined_dataframe = combined_dataframe.with_columns(override_expressions)

        # NOTE: We already know that all columns have the correct dtype
        rules = cls._validation_rules()
        filtered, evaluated = cls._filter_raw(combined_dataframe, rules, cast=False)

        if evaluated is None:
            # When `evaluated` is None, there are no rules and we surely included all
            # items from `remaining_values`
            return filtered, remaining_values, remaining_values.slice(0, 0)

        concat_values = pl.concat([used_values, remaining_values])

        if concat_values.height == 0:
            # If we didn't provide any values, we can simply return empty data frames
            # with the right schema for used and remaining values
            return filtered, concat_values, concat_values

        # NOTE: We can filter `concat_values` using the bitmask from the above filter
        #  operation as the ordering of the custom values is guaranteed to be the
        #  same: `previous_result` and `used_values` contain the same values. Similarly,
        #  `sampled` and `remaining_values` have the same values.
        return (
            filtered,
            concat_values.filter(evaluated.get_column("__final_valid__")),
            concat_values.filter(~evaluated.get_column("__final_valid__")),
        )

    @classmethod
    def _sampling_overrides(cls) -> dict[str, pl.Expr]:
        """Generate expressions to pre-process columns during sampling.

        This method can be overwritten in schemas with complex rules to
        enable sampling data frames in a reasonable number of iterations.

        The provided expressions are applied during sampling after data was generated and
        before it is filtered. In a sampling iteration, only expressions for columns
        that are not defined in the `overrides` argument of that operation are
        pre-processed.

        Returns:
            A dict with entries `column_name: expression`.
        """
        # Do not pre-process any columns by default.
        return {}

    # ---------------------------------- VALIDATION ---------------------------------- #

    @classmethod
    def validate(
        cls, df: pl.DataFrame | pl.LazyFrame, /, *, cast: bool = False
    ) -> DataFrame[Self]:
        """Validate that a data frame satisfies the schema.

        Args:
            df: The data frame to validate.
            cast: Whether columns with a wrong data type in the input data frame are
                cast to the schema's defined data type if possible.

        Returns:
            The (collected) input data frame, wrapped in a generic version of the
            input's data frame type to reflect schema adherence.

        Raises:
            ValidationError: If the input data frame does not satisfy the schema
                definition.

        Note:
            This method _always_ collects the input data frame in order to raise
            potential validation errors.
        """
        # We can dispatch to the `filter` method and raise an error if any row cannot
        # be validated
        df_valid, failures = cls.filter(df, cast=cast)
        if len(failures) > 0:
            raise RuleValidationError(failures.counts())
        return df_valid

    @classmethod
    def is_valid(
        cls, df: pl.DataFrame | pl.LazyFrame, /, *, cast: bool = False
    ) -> bool:
        """Utility method to check whether :meth:`validate` raises an exception.

        Args:
            df: The data frame to check for validity.
            allow_extra_columns: Whether to allow the data frame to contain columns
                that are not defined in the schema.
            cast: Whether columns with a wrong data type in the input data frame are
                cast to the schema's defined data type before running validation. If set
                to ``False``, a wrong data type will result in a return value of
                ``False``.

        Returns:
            Whether the provided dataframe can be validated with this schema.
        """
        try:
            cls.validate(df, cast=cast)
            return True
        except (ValidationError, plexc.InvalidOperationError):
            return False
        except Exception as e:  # pragma: no cover
            raise e

    @classmethod
    def _validate_schema(
        cls,
        lf: pl.LazyFrame,
        *,
        casting: DtypeCasting,
    ) -> pl.LazyFrame:
        schema = lf.collect_schema()
        lf = validate_columns(lf, actual=schema.keys(), expected=cls.column_names())

        if casting == "lenient":
            # Keep around original nullability info to use this information to evaluate
            # whether casting succeeded.
            # NOTE: This code path is only executed from the `filter` method.
            lf = lf.with_columns(
                pl.col(cls.column_names()).is_null().name.suffix(_ORIGINAL_NULL_SUFFIX)
            )

        lf = validate_dtypes(lf, actual=schema, expected=cls.columns(), casting=casting)
        return lf

    # ----------------------------------- FILTERING ---------------------------------- #

    @classmethod
    def filter(
        cls, df: pl.DataFrame | pl.LazyFrame, /, *, cast: bool = False
    ) -> tuple[DataFrame[Self], FailureInfo[Self]]:
        """Filter the data frame by the rules of this schema.

        This method can be thought of as a "soft alternative" to :meth:`validate`.
        While :meth:`validate` raises an exception when a row does not adhere to the
        rules defined in the schema, this method simply filters out these rows and
        succeeds.

        Args:
            df: The data frame to filter for valid rows. The data frame is collected
                within this method, regardless of whether a :class:`~polars.DataFrame`
                or :class:`~polars.LazyFrame` is passed.
            cast: Whether columns with a wrong data type in the input data frame are
                cast to the schema's defined data type if possible. Rows for which the
                cast fails for any column are filtered out.

        Returns:
            A tuple of the validated rows in the input data frame (potentially
            empty) and a simple dataclass carrying information about the rows of the
            data frame which could not be validated successfully.

        Raises:
            ValidationError: If the columns of the input data frame are invalid. This
                happens only if the data frame misses a column defined in the schema or
                a column has an invalid dtype while ``cast`` is set to ``False``.

        Note:
            This method preserves the ordering of the input data frame.
        """
        rules = cls._validation_rules()
        df_filtered, df_evaluated = cls._filter_raw(df, rules, cast)
        if df_evaluated is not None:
            failure_lf = (
                df_evaluated.lazy()
                .filter(~pl.col("__final_valid__"))
                .drop("__final_valid__")
            )
            return (
                df_filtered,
                FailureInfo(lf=failure_lf, rule_columns=list(rules.keys()), schema=cls),
            )
        return df_filtered, FailureInfo(lf=pl.LazyFrame(), rule_columns=[], schema=cls)

    @classmethod
    def _filter_raw(
        cls, df: pl.DataFrame | pl.LazyFrame, rules: dict[str, Rule], cast: bool
    ) -> tuple[DataFrame[Self], pl.DataFrame | None]:
        # First, we check for the schema of the data frame
        lf = cls._validate_schema(df.lazy(), casting=("lenient" if cast else "none"))

        # Then, we filter the data frame
        if cast:
            # Add rules for dtype casting. We can simply check whether the nullability property
            # of any column value changed due to lenient dtype casting (whenever casting fails,
            # the value is set to `null` while previous `null` values are simply kept). To this
            # end, `_validate_schema` kept around the original columns.
            dtype_rules = {
                f"{col}|dtype": Rule(
                    pl.col(col).is_null() == pl.col(f"{col}{_ORIGINAL_NULL_SUFFIX}")
                )
                for col in cls.column_names()
            }
            rules.update(dtype_rules)

        if len(rules) > 0:
            lf_with_eval = with_evaluation_rules(lf, rules)
            if cast:
                # If we cast dtypes, we need to take care of two things:
                # - There's still a bunch of columns showing the original nullability in the
                #   data frame. We simply remove these columns again.
                # - Rules other than the "dtype rule" might not be reliable if type casting
                #   failed, i.e. if the "dtype rule" evaluated to `False`. For this reason,
                #   we set all other rule evaluations to `null` in the case of dtype casting
                #   failure.
                non_dtype_rule_names = [
                    rule for rule in rules if not rule.endswith("|dtype")
                ]
                all_dtype_casts_valid = pl.all_horizontal(pl.col(r"^.*\|dtype$"))
                lf_with_eval = lf_with_eval.drop(
                    cs.matches(f"{_ORIGINAL_NULL_SUFFIX}$")
                ).with_columns(
                    pl.when(all_dtype_casts_valid)
                    .then(pl.col(non_dtype_rule_names))
                    .otherwise(pl.lit(None, dtype=pl.Boolean))
                )

            # At this point, `lf_with_eval` contains the following:
            # - All relevant columns of the original data frame, potentially with cast dtypes
            # - One boolean column for each rule in `rules`
            rule_columns = rules.keys()
            df_evaluated = lf_with_eval.with_columns(
                __final_valid__=pl.all_horizontal(pl.col(rule_columns).fill_null(True))
            ).collect()

            # For the output, partition `lf_evaluated` into the returned data frame `lf`
            # and the invalid data frame
            lf = (
                df_evaluated.lazy()
                .filter(pl.col("__final_valid__"))
                .drop("__final_valid__", *rule_columns)
            )
        else:
            df_evaluated = None

        return (
            lf.collect(),  # type: ignore
            df_evaluated,
        )

    # ------------------------------------ CASTING ----------------------------------- #

    @overload
    @classmethod
    def cast(cls, df: pl.DataFrame, /) -> DataFrame[Self]: ...  # pragma: no cover

    @overload
    @classmethod
    def cast(cls, df: pl.LazyFrame, /) -> LazyFrame[Self]: ...  # pragma: no cover

    @classmethod
    def cast(
        cls, df: pl.DataFrame | pl.LazyFrame, /
    ) -> DataFrame[Self] | LazyFrame[Self]:
        """Cast a data frame to match the schema.

        This method removes superfluous columns and casts all schema columns to the
        correct dtypes. However, it does **not** introspect the data frame contents.

        Hence, this method should be used with care and :meth:`validate` should
        generally be preferred. It is advised to *only* use this method if ``df`` is
        surely known to adhere to the schema.

        Returns:
            The input data frame, wrapped in a generic version of the input's data
            frame type to reflect schema adherence.

        Note:
            If you only require a generic data frame for the type checker, consider
            using :meth:`typing.cast` instead of this method.

        Attention:
            For lazy frames, casting is not performed eagerly. This prevents collecting
            the lazy frame's schema but also means that a call to :meth:`collect`
            further down the line might fail because of the cast and/or missing columns.
        """
        lf = df.lazy().select(
            pl.col(name).cast(col.dtype) for name, col in cls.columns().items()
        )
        if isinstance(df, pl.DataFrame):
            return lf.collect()  # type: ignore
        return lf  # type: ignore

    # --------------------------------- SERIALIZATION -------------------------------- #

    @classmethod
    def serialize(cls) -> str:
        """Serialize this schema to a JSON string.

        Returns:
            The serialized schema.

        Note:
            Serialization within dataframely itself will remain backwards-compatible
            at least within a major version. Until further notice, it will also be
            backwards-compatible across major versions.

        Attention:
            Serialization of :mod:`polars` expressions is not guaranteed to be stable
            across versions of polars. This affects schemas that define custom rules
            or columns with custom checks: a schema serialized with one version of
            polars may not be deserializable with another version of polars.

        Attention:
            This functionality is considered unstable. It may be changed at any time
            without it being considered a breaking change.

        Raises:
            TypeError: If any column contains metadata that is not JSON-serializable.
            ValueError: If any column is not a "native" dataframely column type but
                a custom subclass.
        """
        result = {"versions": serialization_versions(), **cls._as_dict()}
        return json.dumps(result, cls=SchemaJSONEncoder)

    @classmethod
    def _as_dict(cls) -> dict[str, Any]:
        """Return a dictionary representation of this schema.

        This method should only be used internally for the purpose of serializing
        objects referencing schemas.
        """
        return {
            "name": cls.__name__,
            "columns": {
                name: col.as_dict(pl.col(name)) for name, col in cls.columns().items()
            },
            "rules": {
                name: rule.as_dict()
                for name, rule in cls._schema_validation_rules().items()
            },
        }

    # ------------------------------------ PARQUET ----------------------------------- #

    @classmethod
    def write_parquet(
        cls, df: DataFrame[Self], /, file: str | Path | IO[bytes], **kwargs: Any
    ) -> None:
        """Write a typed data frame with this schema to a parquet file.

        This method automatically adds a serialization of this schema to the parquet
        file as metadata. This metadata can be leveraged by :meth:`read_parquet` and
        :meth:`scan_parquet` for more efficient reading, or by external tools.

        Args:
            df: The data frame to write to the parquet file.
            file: The file path or writable file-like object to which to write the
                parquet file. This should be a path to a directory if writing a
                partitioned dataset.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.write_parquet`. ``metadata`` may only be provided if it
                is a dictionary.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`serialize`.
        """
        metadata = kwargs.pop("metadata", {})
        df.write_parquet(
            file, metadata={**metadata, SCHEMA_METADATA_KEY: cls.serialize()}, **kwargs
        )

    @classmethod
    def sink_parquet(
        cls,
        lf: LazyFrame[Self],
        /,
        file: str | Path | IO[bytes] | PartitioningScheme,
        **kwargs: Any,
    ) -> None:
        """Stream a typed lazy frame with this schema to a parquet file.

        This method automatically adds a serialization of this schema to the parquet
        file as metadata. This metadata can be leveraged by :meth:`read_parquet` and
        :meth:`scan_parquet` for more efficient reading, or by external tools.

        Args:
            lf: The lazy frame to write to the parquet file.
            file: The file path, writable file-like object, or partitioning scheme to
                which to write the parquet file.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.write_parquet`. ``metadata`` may only be provided if it
                is a dictionary.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`serialize`.
        """
        metadata = kwargs.pop("metadata", {})
        lf.sink_parquet(
            file, metadata={**metadata, SCHEMA_METADATA_KEY: cls.serialize()}, **kwargs
        )

    @classmethod
    def read_parquet(
        cls,
        source: FileSource,
        *,
        validation: Validation = "warn",
        **kwargs: Any,
    ) -> DataFrame[Self]:
        """Read a parquet file into a typed data frame with this schema.

        Compared to :meth:`polars.read_parquet`, this method checks the parquet file's
        metadata and runs validation if necessary to ensure that the data matches this
        schema.

        Args:
            source: Path, directory, or file-like object from which to read the data.
            validation: The strategy for running validation when reading the data:

                - ``"allow"`: The method tries to read the parquet file's metadata. If
                  the stored schema matches this schema, the data frame is read without
                  validation. If the stored schema mismatches this schema or no schema
                  information can be found in the metadata, this method automatically
                  runs :meth:`validate` with ``cast=True``.
                - ``"warn"`: The method behaves similarly to ``"allow"``. However,
                  it prints a warning if validation is necessary.
                - ``"forbid"``: The method never runs validation automatically and only
                  returns if the schema stored in the parquet file's metadata matches
                  this schema.
                - ``"skip"``: The method never runs validation and simply reads the
                  parquet file, entrusting the user that the schema is valid. _Use this
                  option carefully and consider replacing it with
                  :meth:`polars.read_parquet` to convey the purpose better_.

            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.read_parquet`.

        Returns:
            The data frame with this schema.

        Raises:
            ValidationRequiredError: If no schema information can be read from the
                source and ``validation`` is set to ``"forbid"``.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`serialize`.
        """
        if not cls._requires_validation_for_reading_parquet(source, validation):
            return pl.read_parquet(source, **kwargs)  # type: ignore
        return cls.validate(pl.read_parquet(source, **kwargs), cast=True)

    @classmethod
    def scan_parquet(
        cls,
        source: FileSource,
        *,
        validation: Validation = "warn",
        **kwargs: Any,
    ) -> LazyFrame[Self]:
        """Lazily read a parquet file into a typed data frame with this schema.

        Compared to :meth:`polars.scan_parquet`, this method checks the parquet file's
        metadata and runs validation if necessary to ensure that the data matches this
        schema.

        Args:
            source: Path, directory, or file-like object from which to read the data.
            validation: The strategy for running validation when reading the data:

                - ``"allow"`: The method tries to read the parquet file's metadata. If
                  the stored schema matches this schema, the data frame is read without
                  validation. If the stored schema mismatches this schema or no schema
                  information can be found in the metadata, this method automatically
                  runs :meth:`validate` with ``cast=True``.
                - ``"warn"`: The method behaves similarly to ``"allow"``. However,
                  it prints a warning if validation is necessary.
                - ``"forbid"``: The method never runs validation automatically and only
                  returns if the schema stored in the parquet file's metadata matches
                  this schema.
                - ``"skip"``: The method never runs validation and simply reads the
                  parquet file, entrusting the user that the schema is valid. _Use this
                  option carefully and consider replacing it with
                  :meth:`polars.scan_parquet` to convey the purpose better_.

            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.scan_parquet`.

        Returns:
            The data frame with this schema.

        Raises:
            ValidationRequiredError: If no schema information can be read from the
                source and ``validation`` is set to ``"forbid"``.

        Note:
            Due to current limitations in dataframely, this method actually reads the
            parquet file into memory if ``validation`` is ``"warn"`` or ``"allow"`` and
            validation is required.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`serialize`.
        """
        if not cls._requires_validation_for_reading_parquet(source, validation):
            return pl.scan_parquet(source, **kwargs)  # type: ignore
        return cls.validate(pl.read_parquet(source, **kwargs), cast=True).lazy()

    @classmethod
    def _requires_validation_for_reading_parquet(
        cls, source: FileSource, validation: Validation
    ) -> bool:
        if validation == "skip":
            return False

        # First, we check whether the source provides the dataframely schema. If it
        # does, we check whether it matches this schema. If it does, we assume that the
        # data adheres to the schema and we do not need to run validation.
        serialized_schema = (
            read_parquet_metadata_schema(source)
            if not isinstance(source, list)
            else None
        )
        if serialized_schema is not None:
            if cls.matches(serialized_schema):
                return False

        # Otherwise, we definitely need to run validation. However, we emit different
        # information to the user depending on the value of `validate`.
        msg = (
            "current schema does not match stored schema"
            if serialized_schema is not None
            else "no schema to check validity can be read from the source"
        )
        if validation == "forbid":
            raise ValidationRequiredError(
                f"Cannot read parquet file from '{source!r}' without validation: {msg}."
            )
        if validation == "warn":
            warnings.warn(
                f"Reading parquet file from '{source!r}' requires validation: {msg}."
            )
        return True

    # ----------------------------- THIRD-PARTY PACKAGES ----------------------------- #

    @classmethod
    def polars_schema(cls) -> pl.Schema:
        """Obtain the polars schema for this schema.

        Returns:
            A :mod:`polars` schema that mirrors the schema defined by this class.
        """
        return pl.Schema({name: col.dtype for name, col in cls.columns().items()})

    @classmethod
    def sql_schema(cls, dialect: sa.Dialect) -> list[sa.Column]:
        """Obtain the SQL schema for a particular dialect for this schema.

        Args:
            dialect: The dialect for which to obtain the SQL schema. Note that column
                datatypes may differ across dialects.

        Returns:
            A list of :mod:`sqlalchemy` columns that can be used to create a table
            with the schema as defined by this class.
        """
        return [
            col.sqlalchemy_column(name, dialect) for name, col in cls.columns().items()
        ]

    @classmethod
    def pyarrow_schema(cls) -> pa.Schema:
        """Obtain the pyarrow schema for this schema.

        Returns:
            A :mod:`pyarrow` schema that mirrors the schema defined by this class.
        """
        return pa.schema(
            [col.pyarrow_field(name) for name, col in cls.columns().items()]
        )

    # ----------------------------------- EQUALITY ----------------------------------- #

    @classmethod
    def matches(cls, other: type[Schema]) -> bool:
        """Check whether this schema semantically matches another schema.

        This method checks whether the schemas have the same columns (with the same
        data types and constraints) as well as the same rules.

        Args:
            other: The schema to compare with.

        Returns:
            Whether the schemas are semantically equal.
        """

        def _columns_match(lhs: dict[str, Column], rhs: dict[str, Column]) -> bool:
            if lhs.keys() != rhs.keys():
                return False
            return all(lhs[name].matches(rhs[name], pl.col(name)) for name in lhs)

        def _rules_match(lhs: dict[str, Rule], rhs: dict[str, Rule]) -> bool:
            if lhs.keys() != rhs.keys():
                return False
            return all(lhs[name].matches(rhs[name]) for name in lhs)

        return _columns_match(cls.columns(), other.columns()) and _rules_match(
            cls._schema_validation_rules(), other._schema_validation_rules()
        )


def read_parquet_metadata_schema(
    source: str | Path | IO[bytes] | bytes,
) -> type[Schema] | None:
    """Read a dataframely schema from the metadata of a parquet file.

    Args:
        source: Path to a parquet file or a file-like object that contains the metadata.

    Returns:
        The schema that was serialized to the metadata or ``None`` if no schema metadata
        is found.
    """
    metadata = pl.read_parquet_metadata(source)
    if (schema_metadata := metadata.get(SCHEMA_METADATA_KEY)) is not None:
        return deserialize_schema(schema_metadata)
    return None


def deserialize_schema(data: str) -> type[Schema]:
    """Deserialize a schema from a JSON string.

    This method allows to dynamically load a schema from its serialization, without
    having to know the schema to load in advance.

    Args:
        data: The JSON string created via :meth:`Schema.serialize`.

    Returns:
        The schema loaded from the JSON data.

    Raises:
        ValueError: If the schema format version is not supported.

    Attention:
        This functionality is considered unstable. It may be changed at any time
        without it being considered a breaking change.

    See also:
        :meth:`Schema.serialize` for additional information on serialization.
    """
    decoded = json.loads(data, cls=SchemaJSONDecoder)
    if (format := decoded["versions"]["format"]) != SERIALIZATION_FORMAT_VERSION:
        raise ValueError(f"Unsupported schema format version: {format}")
    return _schema_from_dict(decoded)


def _schema_from_dict(data: dict[str, Any]) -> type[Schema]:
    """Create a schema from a dictionary representation.

    This function should only be used internally for the purpose of deserializing
    objects referencing schemas.
    """
    return type(
        f"{data['name']}_dynamic",
        (Schema,),
        {
            **{name: column_from_dict(col) for name, col in data["columns"].items()},
            **{name: rule_from_dict(rule) for name, rule in data["rules"].items()},
        },
    )
