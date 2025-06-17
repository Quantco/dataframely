# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import polars as pl

ValidationFunction = Callable[[], pl.Expr]
# For whatever reason, a method annotated with `@classmethod` will have type
# `Callable[[type[...]], ...]`, but its value will not be callable but
# an instance of `classmethod`. To make mypy happy we need both types here.
LazyValidationFunction = (
    ValidationFunction | Callable[[type[Any]], pl.Expr] | classmethod
)


class Rule:
    """Internal class representing validation rules."""

    def __init__(self, expr_or_validation_fn: pl.Expr | LazyValidationFunction) -> None:
        self.expr_or_validation_fn = expr_or_validation_fn


class GroupRule(Rule):
    """Rule that is evaluated on a group of columns."""

    def __init__(
        self,
        expr_or_validation_fn: pl.Expr | LazyValidationFunction,
        group_columns: list[str],
    ) -> None:
        super().__init__(expr_or_validation_fn)
        self.group_columns = group_columns


def rule(
    *, group_by: list[str] | None = None, lazy: bool = False
) -> Callable[[LazyValidationFunction], Rule]:
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
        lazy: If set to ``True``, the rule will be evaluated lazily. This means that the
            dtype of the returned expression is not validated to be a boolean expression!
            Use this if you need to access the schema class in your rule function, want
            to use `@classmethod` on the rule or need to access constants defined after
            the schema.

    Note:
        You'll need to explicitly handle ``null`` values in your columns when defining
        rules. By default, any rule that evaluates to ``null`` because one of the
        columns used in the rule is ``null`` is interpreted as ``true``, i.e. the row
        is assumed to be valid.
    """

    def decorator(validation_fn: LazyValidationFunction) -> Rule:
        if isinstance(validation_fn, classmethod) and not lazy:
            raise ValueError(
                "Using `@classmethod` on a rule requires `lazy=True` to be set."
            )

        if group_by is not None:
            # mypy doesnt understand that we only call validation_fn when its not a class method.
            return GroupRule(
                expr_or_validation_fn=validation_fn if lazy else validation_fn(),  # type: ignore[call-arg, operator]
                group_columns=group_by,
            )
        return Rule(expr_or_validation_fn=validation_fn if lazy else validation_fn())  # type: ignore[call-arg, operator]

    return decorator


def lazy_rule(
    *, group_by: list[str] | None = None
) -> Callable[[LazyValidationFunction], Rule]:
    """Mark a function as a rule to evaluate during validation.

    This does the same as `rule`, but allows for lazy evaluation of rules. This means the
    dtype of the returned expression is not validated to be a boolean expression! Use this
    method if you need to access the schema class in your rule function, want to use
    `@classmethod` on the rule or need to access constants defined after the schema.

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

    def decorator(validation_fn: LazyValidationFunction) -> Rule:
        if group_by is not None:
            return GroupRule(
                expr_or_validation_fn=validation_fn, group_columns=group_by
            )
        return Rule(expr_or_validation_fn=validation_fn)

    return decorator


# ------------------------------------------------------------------------------------ #
#                                      EVALUATION                                      #
# ------------------------------------------------------------------------------------ #


def _call_lazy_rule(
    validation_fn: LazyValidationFunction, schema_class: type
) -> pl.Expr:
    if isinstance(validation_fn, classmethod):
        return validation_fn.__func__(schema_class)

    # As written above, `@classmethod` annotation will make the type (to the type checker)
    # look like a `Callable[[type[...]], ...]`, but the value will not be callable.
    # Hence we can ignore the missing argument type error here.
    return validation_fn()  # type: ignore[call-arg]


def with_evaluation_rules(
    lf: pl.LazyFrame, rules: dict[str, Rule], schema_class: type | None = None
) -> pl.LazyFrame:
    """Add evaluations of a set of rules on a data frame.

    Args:
        lf: The data frame on which to evaluate the rules.
        rules: The rules to evaluate where the key of the dictionary provides the name
            of the rule.
        schema_class: The schema class to use for lazy evaluation of rules. If this is
            not provided, rules that are not already expressed as `pl.Expr` will not be
            evaluated.
    Returns:
        The input lazy frame along with one boolean column for each rule with the name
        of the rule. For each rule, a value of ``True`` indicates successful validation
        while ``False`` indicates an issue.
    """
    # Rules must be distinguished into two types of rules:
    #  1. Simple rules can simply be selected on the data frame
    #  2. "Group" rules require a `group_by` and a subsequent join
    simple_exprs = {
        name: rule.expr_or_validation_fn
        if isinstance(rule.expr_or_validation_fn, pl.Expr)
        # Mypy doesnt understand we only call this with schema_class != None
        else _call_lazy_rule(rule.expr_or_validation_fn, schema_class)  # type: ignore[arg-type]
        for name, rule in rules.items()
        if not isinstance(rule, GroupRule)
        and (
            schema_class is not None or isinstance(rule.expr_or_validation_fn, pl.Expr)
        )
    }
    group_rules = {
        name: rule for name, rule in rules.items() if isinstance(rule, GroupRule)
    }

    # Before we can select all of the simple expressions, we need to turn the
    # group rules into something to use in a `select` statement as well.
    return (
        # NOTE: A value of `null` always validates successfully as nullability should
        #  already be checked via dedicated rules.
        _with_group_rules(lf, group_rules, schema_class).with_columns(
            **{name: expr.fill_null(True) for name, expr in simple_exprs.items()},
        )
    )


def _with_group_rules(
    lf: pl.LazyFrame, rules: dict[str, GroupRule], schema_class: type | None
) -> pl.LazyFrame:
    # First, we partition the rules by group columns. This will minimize the number
    # of `group_by` calls and joins to make.
    grouped_rules: dict[frozenset[str], dict[str, pl.Expr]] = defaultdict(dict)
    for name, rule in rules.items():
        if schema_class is None and not isinstance(rule.expr_or_validation_fn, pl.Expr):
            continue
        # NOTE: `null` indicates validity, see note above.
        grouped_rules[frozenset(rule.group_columns)][name] = (
            rule.expr_or_validation_fn
            if isinstance(rule.expr_or_validation_fn, pl.Expr)
            # Mypy doesnt understand we only call this with schema_class != None
            else _call_lazy_rule(rule.expr_or_validation_fn, schema_class)  # type: ignore[arg-type]
        ).fill_null(True)

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
