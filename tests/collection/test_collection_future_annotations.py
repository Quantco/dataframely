# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import (
    annotations,  # This must not exist in files that define collections
)

import pytest

import dataframely as dy
from dataframely.exc import AnnotationImplementationError


class MySchema(dy.Schema):
    a = dy.Integer()


def test_collection_future_annotations() -> None:
    with pytest.raises(AnnotationImplementationError) as exception:

        class MyCollection(dy.Collection):
            member: dy.LazyFrame[MySchema]

    assert "`from __future__ import annotations`" in str(exception.value)
