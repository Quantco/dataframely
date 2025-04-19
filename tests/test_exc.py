# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

from dataframely.exc import DtypeValidationError, RuleValidationError, ValidationError


def test_validation_error_str() -> None:
    message = "validation failed"
    exc = ValidationError(message)
    assert str(exc) == message


def test_dtype_validation_error_str() -> None:
    exc = DtypeValidationError(
        errors={"a": (pl.Int64, pl.String), "b": (pl.Boolean, pl.String)}
    )
    assert str(exc).split("\n") == [
        "2 columns have an invalid dtype:",
        " - 'a': got dtype 'Int64' but expected 'String'",
        " - 'b': got dtype 'Boolean' but expected 'String'",
    ]


def test_rule_validation_error_str() -> None:
    exc = RuleValidationError(
        {
            "b|max_length": 1500,
            "a|nullability": 2,
            "primary_key": 2000,
            "a|min_length": 5,
        },
    )
    assert str(exc).split("\n") == [
        "4 rules failed validation:",
        " - 'primary_key' failed validation for 2,000 rows",
        " * Column 'a' failed validation for 2 rules:",
        "   - 'min_length' failed for 5 rows",
        "   - 'nullability' failed for 2 rows",
        " * Column 'b' failed validation for 1 rules:",
        "   - 'max_length' failed for 1,500 rows",
    ]
