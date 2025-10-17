# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import dataframely as dy


@pytest.mark.parametrize("column_type", [dy.Int64, dy.String, dy.Float32, dy.Decimal])
def test_no_nullable_primary_key(column_type: type[dy.Column]) -> None:
    with pytest.raises(ValueError):
        column_type(primary_key=True, nullable=True)
