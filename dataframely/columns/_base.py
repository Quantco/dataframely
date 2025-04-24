# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

# ------------------------------------------------------------------------------------ #
#                                        COLUMNS                                       #
# ------------------------------------------------------------------------------------ #


class Column(ABC):
    """Abstract base class for data frame column definitions.

    This class is merely supposed to be used in :class:`~dataframely.Schema`
    definitions.
    """

    def __init__(
        self,
        *,
        nullable: bool | None = None,
        primary_key: bool = False,
        check: Callable[[pl.Expr], pl.Expr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
                If `None`, the default behavior is as follows:
                - If `primary_key` is `True`, `nullable` defaults to `False`.
                - If `primary_key` is `False`, `nullable` defaults to `True`.
            primary_key: Whether this column is part of the primary key of the schema.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name. If unset, dataframely
                internally sets the alias to the column's name in the parent schema.
            metadata: A dictionary of metadata to attach to the column.
        """
        self.nullable = nullable if nullable is not None else not primary_key
        self.primary_key = primary_key
        self.check = check
        self.alias = alias
        self.metadata = metadata

    # ------------------------------------- DTYPE ------------------------------------ #

    @property
    @abstractmethod
    def dtype(self) -> pl.DataType:
        """The :mod:`polars` dtype equivalent of this column definition's data type.

        This is primarily used for creating empty data frames with an appropriate
        schema. Thus, it should describe the default dtype equivalent if this data type
        encompasses multiple underlying data types.
        """

    def validate_dtype(self, dtype: PolarsDataType) -> bool:
        """Validate if the :mod:`polars` data type satisfies the column definition.

        Args:
            dtype: The dtype to validate.

        Returns:
            Whether the dtype is valid.
        """
        return self.dtype == dtype

    # ---------------------------------- VALIDATION ---------------------------------- #

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        """A set of rules evaluating whether a data frame column satisfies the column's
        constraints.

        Args:
            expr: An expression referencing the column of the data frame, i.e. an
                expression created by calling :meth:`polars.col`.

        Returns:
            A mapping from validation rule names to expressions that provide exactly
            one boolean value per column item indicating whether validation with respect
            to the rule is successful. A value of ``False`` indicates invalid data, i.e.
            unsuccessful validation.
        """
        result = {}
        if not self.nullable:
            result["nullability"] = expr.is_not_null()
        if self.check is not None:
            result["check"] = self.check(expr)
        return result

    # -------------------------------------- SQL ------------------------------------- #

    def sqlalchemy_column(self, name: str, dialect: sa.Dialect) -> sa.Column:
        """Obtain the SQL column specification of this column definition.

        Args:
            name: The name of the column.
            dialect: The SQL dialect for which to generate the column specification.

        Returns:
            The column as specified in :mod:`sqlalchemy`.
        """
        return sa.Column(
            name,
            self.sqlalchemy_dtype(dialect),
            nullable=self.nullable,
            primary_key=self.primary_key,
            autoincrement=False,
        )

    @abstractmethod
    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        """The :mod:`sqlalchemy` dtype equivalent of this column data type."""

    # ------------------------------------ PYARROW ----------------------------------- #

    def pyarrow_field(self, name: str) -> pa.Field:
        """Obtain the pyarrow field of this column definition.

        Args:
            name: The name of the column.

        Returns:
            The :mod:`pyarrow` field definition.
        """
        return pa.field(name, self.pyarrow_dtype, nullable=self.nullable)

    @property
    @abstractmethod
    def pyarrow_dtype(self) -> pa.DataType:
        """The :mod:`pyarrow` dtype equivalent of this column data type."""

    # ------------------------------------ HELPER ------------------------------------ #

    @property
    def col(self) -> pl.Expr:
        """Obtain a Polars column expression for the column."""
        if self.alias is None:
            raise ValueError("Cannot obtain column expression if alias is ``None``.")
        return pl.col(self.alias)

    # ----------------------------------- SAMPLING ----------------------------------- #

    def sample(self, generator: Generator, n: int = 1) -> pl.Series:
        """Sample random elements adhering to the constraints of this column.

        Args:
            generator: The generator to use for sampling elements.
            n: The number of elements to sample.

        Returns:
            A series with the predefined number of elements. All elements are guaranteed
            to adhere to the column's constraints.

        Raises:
            ValueError: If this column has a custom check. In this case, random values
                cannot be guaranteed to adhere to the column's constraints while
                providing any guarantees on the computational complexity.
        """
        if self.check is not None:
            raise ValueError(
                "Samples cannot be generated for columns with custom checks."
            )
        return self._sample_unchecked(generator, n)

    @abstractmethod
    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        """Private method sampling random elements without checking for custom
        checks."""

    @property
    def _null_probability(self) -> float:
        """Private utility for the null probability used during sampling."""
        return 0.1 if self.nullable else 0

    # -------------------------------- DUNDER METHODS -------------------------------- #

    def __str__(self) -> str:
        return self.__class__.__name__.lower()
