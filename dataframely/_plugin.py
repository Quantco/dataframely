# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from pathlib import Path
from typing import TypeAlias

import polars as pl
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parent

IntoExpr: TypeAlias = pl.Expr | str


def all_rules_horizontal(rules: IntoExpr | Iterable[IntoExpr]) -> pl.Expr:
    """Execute :mod:`~polars.all_horizontal` for a set of rules.

    This implementation is more efficient and yields better errors than
    :mod:`~polars.all_horizontal`.

    Args:
        rules: The rules to evaluate.

    Returns:
        A boolean expression with one result per row.
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="all_rules_horizontal",
        args=rules,
        use_abs_path=True,
    )


def all_rules(rules: IntoExpr | Iterable[IntoExpr]) -> pl.Expr:
    """Execute :mod:`~polars.all_horizontal` and `.all` for a set of rules.

    This is more efficient than running the two operations one after the other.

    Args:
        rules: The rules to evaluate.

    Returns:
        A scalar boolean expression.
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="all_rules",
        args=rules,
        use_abs_path=True,
        returns_scalar=True,
    )


def all_rules_required(
    rules: IntoExpr | Iterable[IntoExpr],
    *,
    null_is_valid: bool = True,
    schema_name: str,
    data_columns: Iterable[IntoExpr] | None = None,
) -> pl.Expr:
    """Execute :mod:`~polars.all_horizontal` and `.all` for a set of rules.

    Contrary to :meth:`all_rules`, this method raises a
    :mod:`~polars.exceptions.ComputeError` at execution time if any rule indicates a
    validation failure. The `ComputeError` includes a helpful error message.

    Args:
        rules: The rules to evaluate.
        schema_name: The name of the schema being validated. This is used to produce
            better error messages.
        null_is_valid: Whether to treat null values as valid (i.e., `true`).
        data_columns: Optional data columns to include for generating example rows in
            error messages. If provided, up to 5 distinct example rows are included
            for each failing rule.

    Returns:
        A scalar boolean expression.
    """
    rules_list = [rules] if isinstance(rules, pl.Expr) else list(rules)
    num_rule_columns = len(rules_list)
    data_columns_list = list(data_columns) if data_columns is not None else []
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="all_rules_required",
        args=[*rules_list, *data_columns_list],
        kwargs={
            "null_is_valid": null_is_valid,
            "schema_name": schema_name,
            "num_rule_columns": num_rule_columns,
        },
        use_abs_path=True,
        returns_scalar=True,
    )
