# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import pytest

from dataframely._storage._fsspec import get_file_prefix


class MockFS:
    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol


def test_get_file_prefix() -> None:
    assert get_file_prefix(MockFS("file")) == ""
    assert get_file_prefix(MockFS("s3")) == "s3://"
    assert get_file_prefix(MockFS(["file", "whatever"])) == ""
    assert get_file_prefix(MockFS(["s3", "whatever"])) == "s3://"


@pytest.mark.parametrize("protocol", [5, None, [5]])
def test_get_file_prefix_invalid(protocol: Any) -> None:
    with pytest.raises(ValueError):
        assert 1 == get_file_prefix(MockFS(protocol))
