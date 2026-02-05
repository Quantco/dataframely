# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from ._base import StorageBackend
from ._fsspec import get_file_prefix

__all__ = [
    "StorageBackend",
    "get_file_prefix"
]
