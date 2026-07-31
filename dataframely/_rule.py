# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import polars as pl

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

ValidationFunction = Callable[[Any], pl.Expr]


class Rule:
    """Internal class representing validation rules."""

    def __init__(self, expr: pl.Expr | Callable[[], pl.Expr]) -> None:
        self._expr = expr

    @property
    def expr(self) -> pl.Expr:
        """Get the expression of the rule."""
        if callable(self._expr):
            return self._expr()
        return self._expr

    def matches(self, other: Rule) -> bool:
        """Check whether this rule semantically matches another rule.

        Args:
            other: The rule to compare with.

        Returns:
            Whether the rules are semantically equal.
        """
        return self.expr.meta.eq(other.expr)

    def __repr__(self) -> str:
        return str(self.expr)


class DtypeCastRule(Rule):
    """Rule that evaluates whether casting a column to another dtype is successful.

    The only purpose of this rule is to provide a runtime type to distinguish it from
    other rules.
    """


# -------------------------------------- FACTORY ------------------------------------- #


class RuleFactory:
    """Factory class for rules created within schemas."""

    def __init__(self, validation_fn: Callable[[Any], pl.Expr]) -> None:
        self.validation_fn = validation_fn

    @classmethod
    def from_rule(cls, rule: Rule) -> Self:
        """Create a rule factory from an existing rule."""
        return cls(validation_fn=lambda _: rule.expr)

    def make(self, schema: Any) -> Rule:
        """Create a new rule from this factory."""
        return Rule(expr=lambda: self.validation_fn(schema))


def rule() -> Callable[[ValidationFunction], RuleFactory]:
    """Mark a function as a rule to evaluate during validation.

    The name of the function will be used as the name of the rule. The function should
    return an expression providing a boolean value whether a row is valid wrt. the rule.
    A value of `true` indicates validity.

    Rules should be used only in the following two circumstances:

    - Validation requires accessing multiple columns (e.g. if valid values of column A
      depend on the value in column B).
    - Validation must be performed on groups of rows (e.g. if a column A must not
      contain any duplicate values among rows with the same value in column B). This
      can be achieved with an `over` expression.

    In all other instances, column-level validation rules should be preferred as it aids
    readability and improves error messages.

    Note:
        You'll need to explicitly handle `null` values in your columns when defining
        rules. By default, any rule that evaluates to `null` because one of the
        columns used in the rule is `null` is interpreted as `true`, i.e. the row
        is assumed to be valid.
    """

    def decorator(validation_fn: ValidationFunction) -> RuleFactory:
        return RuleFactory(validation_fn=validation_fn)

    return decorator


# ------------------------------------------------------------------------------------ #
#                                      EVALUATION                                      #
# ------------------------------------------------------------------------------------ #


def with_evaluation_rules(lf: pl.LazyFrame, rules: dict[str, Rule]) -> pl.LazyFrame:
    """Add evaluations of a set of rules on a data frame.

    Args:
        lf: The data frame on which to evaluate the rules.
        rules: The rules to evaluate where the key of the dictionary provides the name
            of the rule.

    Returns:
        The input lazy frame along with one boolean column for each rule with the name
        of the rule. For each rule, a value of `True` indicates successful validation
        while `False` indicates an issue.
    """
    exprs = {name: rule.expr for name, rule in rules.items()}
    result = (
        # NOTE: A value of `null` always validates successfully as nullability should
        #  already be checked via dedicated rules.
        lf.with_columns(
            **{name: expr.fill_null(True) for name, expr in exprs.items()},
        )
    )

    # If there is at least one rule that checks for successful dtype casting, we need
    # to take an extra step: rules other than the "dtype rules" might not be reliable
    # if casting failed, i.e. if any of the "dtype rules" evaluated to `False`. For
    # this reason, we set all other rule evaluations to `null` in the case of dtype
    # casting failure.
    dtype_rule_names = [
        name for name, rule in rules.items() if isinstance(rule, DtypeCastRule)
    ]
    if len(dtype_rule_names) > 0:
        non_dtype_rule_names = [
            name for name, rule in rules.items() if not isinstance(rule, DtypeCastRule)
        ]
        all_dtype_casts_valid = pl.all_horizontal(dtype_rule_names)
        return result.with_columns(
            pl.when(all_dtype_casts_valid)
            .then(pl.col(non_dtype_rule_names))
            .otherwise(pl.lit(None, dtype=pl.Boolean))
        )

    return result
