# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections import defaultdict
from collections.abc import Callable

import polars as pl

ValidationFunction = Callable[[], pl.Expr]


class Rule:
    """Internal class representing validation rules."""

    def __init__(self, expr: pl.Expr) -> None:
        self.expr = expr


class GroupRule(Rule):
    """Rule that is evaluated on a group of columns."""

    def __init__(self, expr: pl.Expr, group_columns: list[str]) -> None:
        super().__init__(expr)
        self.group_columns = group_columns


def rule(*, group_by: list[str] | None = None) -> Callable[[ValidationFunction], Rule]:
    """Mark a function as a rule to evaluate during validation.

    The name of the function will be used as the name of the rule. The function should
    return an expression providing a boolean value whether a row is valid wrt. the rule.
    A value of ``true`` indicates validity.

    Rules should be used only in the following two circumstances:

    - Validation requires accessing multiple columns (e.g. if valid values of column A
      depend on the value in column B).
    - Validation must be performed on groups of rows (e.g. if a column A must not
      contain any duplicate values among rows with the same value in column B).

    In all other instances, column-level validation rules should be preferred as it aids
    readability and improves error messages.

    Args:
        group_by: An optional list of columns to group by for rules operating on groups
            of rows. If this list is provided, the returned expression must return a
            single boolean value, i.e. some kind of aggregation function must be used
            (e.g. ``sum``, ``any``, ...).

    Note:
        You'll need to explicitly handle ``null`` values in your columns when defining
        rules. By default, any rule that evaluates to ``null`` because one of the
        columns used in the rule is ``null`` is interpreted as ``true``, i.e. the row
        is assumed to be valid.
    """

    def decorator(validation_fn: ValidationFunction) -> Rule:
        if group_by is not None:
            return GroupRule(expr=validation_fn(), group_columns=group_by)
        return Rule(expr=validation_fn())

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
        of the rule. For each rule, a value of ``True`` indicates successful validation
        while ``False`` indicates an issue.
    """
    # Rules must be distinguished into two types of rules:
    #  1. Simple rules can simply be selected on the data frame
    #  2. "Group" rules require a `group_by` and a subsequent join
    simple_exprs = {
        name: rule.expr
        for name, rule in rules.items()
        if not isinstance(rule, GroupRule)
    }
    group_rules = {
        name: rule for name, rule in rules.items() if isinstance(rule, GroupRule)
    }

    # Before we can select all of the simple expressions, we need to turn the
    # group rules into something to use in a `select` statement as well.
    return (
        # NOTE: A value of `null` always validates successfully as nullability should
        #  already be checked via dedicated rules.
        _with_group_rules(lf, group_rules).with_columns(
            **{name: expr.fill_null(True) for name, expr in simple_exprs.items()},
        )
    )


def _with_group_rules(lf: pl.LazyFrame, rules: dict[str, GroupRule]) -> pl.LazyFrame:
    # First, we partition the rules by group columns. This will minimize the number
    # of `group_by` calls and joins to make.
    grouped_rules: dict[frozenset[str], dict[str, pl.Expr]] = defaultdict(dict)
    for name, rule in rules.items():
        # NOTE: `null` indicates validity, see note above.
        grouped_rules[frozenset(rule.group_columns)][name] = rule.expr.fill_null(True)

    # Then, for each `group_by`, we apply the relevant rules and keep all the rule
    # evaluations around
    group_evaluations: dict[frozenset[str], pl.LazyFrame] = {}
    for group_columns, group_rules in grouped_rules.items():
        # We group by the group columns and apply all expressions
        group_evaluations[group_columns] = lf.group_by(group_columns).agg(**group_rules)

    # Eventually, we apply the rule evaluations onto the input data frame. For this,
    # we're using left-joins. This has two effects:
    #  - We're essentially "broadcasting" the results within each group across rows
    #    in the same group.
    #  - While an inner-join would be semantically more accurate, the left-join
    #    preserves the order of the left data frame.
    result = lf
    for group_columns, frame in group_evaluations.items():
        result = result.join(
            frame, on=list(group_columns), how="left", nulls_equal=True
        )
    return result
