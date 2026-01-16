# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause


def assert_failure_info_metadata(metadata: str | None) -> str:
    if metadata:
        return metadata
    raise ValueError(
        "The required FailureInfo metadata was not found in the storage backend."
    )
