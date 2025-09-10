# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


import pytest
from polars.testing import assert_frame_equal

import dataframely as dy

pytestmark = pytest.mark.s3


class MyFirstSchema(dy.Schema):
    a = dy.UInt16(primary_key=True)


class MySecondSchema(dy.Schema):
    x = dy.UInt16(primary_key=True)
    y = dy.Integer()


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema] | None


# ------------------------------------------------------------------------------------ #


def test_read_write_parquet(s3_tmp_path: str) -> None:
    # Arrange
    collection = MyCollection.sample(100)
    collection.write_parquet(s3_tmp_path)

    # Act
    breakpoint()
    read_collection = MyCollection.read_parquet(s3_tmp_path)

    # Assert
    assert_frame_equal(collection.first, read_collection.first)
    assert collection.second is not None
    assert read_collection.second is not None
    assert_frame_equal(collection.second, read_collection.second)
