# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

from dataframely._rule import Rule, with_evaluation_rules


def rules_from_exprs(exprs: dict[str, pl.Expr]) -> dict[str, Rule]:
    """Turn a set of expressions into simple rules.

    Args:
        exprs: The expressions, mapping from names to :class:`polars.Expr`.

    Returns:
        The rules corresponding to the expressions.
    """
    return {name: Rule(expr) for name, expr in exprs.items()}


def evaluate_rules(lf: pl.LazyFrame, rules: dict[str, Rule]) -> pl.LazyFrame:
    """Evaluate the provided rules and return the rules' evaluation.

    Args:
        lf: The data frame on which to evaluate the rules.
        rules: The rules to evaluate where the key of the dictionary provides the name
            of the rule.

    Returns:
        The same return value as :meth:`with_evaluation_rules` only that the columns
        of the input data frame are dropped.
    """
    return with_evaluation_rules(lf, rules).drop(lf.collect_schema().keys())
