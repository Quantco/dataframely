# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from typing import Any

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely._deprecation import warn_nullable_default_change
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
        check: (
            Callable[[pl.Expr], pl.Expr]
            | list[Callable[[pl.Expr], pl.Expr]]
            | dict[str, Callable[[pl.Expr], pl.Expr]]
            | None
        ) = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            check: A custom rule or multiple rules to run for this column. This can be:
                - A single callable that returns a non-aggregated boolean expression.
                The name of the rule is derived from the callable name, or defaults to
                "check" for lambdas.
                - A list of callables, where each callable returns a non-aggregated
                boolean expression. The name of the rule is derived from the callable
                name, or defaults to "check" for lambdas. Where multiple rules result
                in the same name, the suffix __i is appended to the name.
                - A dictionary mapping rule names to callables, where each callable
                returns a non-aggregated boolean expression.
                All rule names provided here are given the prefix "check_".
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name. If unset, dataframely
                internally sets the alias to the column's name in the parent schema.
            metadata: A dictionary of metadata to attach to the column.
        """
        if nullable is None:
            warn_nullable_default_change()
            nullable = True

        self.nullable = nullable and not primary_key
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
            if isinstance(self.check, dict):
                for rule_name, rule_callable in self.check.items():
                    result[f"check__{rule_name}"] = rule_callable(expr)
            else:
                list_of_rules = (
                    self.check if isinstance(self.check, list) else [self.check]
                )
                # Get unique names for rules from callables
                rule_names = self._derive_check_rule_names(list_of_rules)
                for rule_name, rule_callable in zip(rule_names, list_of_rules):
                    result[rule_name] = rule_callable(expr)

        return result

    def _derive_check_rule_names(
        self, rules: list[Callable[[pl.Expr], pl.Expr]]
    ) -> list[str]:
        """Generate unique names for rule callables.

        For callables with the same name, appends a suffix __i where i is the index
        of occurrence (starting from 0), but only if there are duplicates.

        Args:
            rules: List of rule callables.

        Returns:
            List of unique names corresponding to the rule callables.
        """
        base_names = [
            f"check__{rule.__name__}" if rule.__name__ != "<lambda>" else "check"
            for rule in rules
        ]

        # Count occurrences using Counter
        name_counts = Counter(base_names)

        # Append suffixes to names that are duplicated
        final_names = []
        duplicate_counter: dict[str, int] = {
            name: 0 for name in name_counts if name_counts[name] > 1
        }
        for name in base_names:
            if name_counts[name] > 1:
                postfix = duplicate_counter[name]
                final_names.append(f"{name}__{postfix}")
                duplicate_counter[name] += 1
            else:
                final_names.append(name)

        return final_names

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
