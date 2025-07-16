# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import TypeVar

from cloudpathlib import CloudPath
from cloudpathlib.local.localpath import LocalPath

T = TypeVar("T")


def handle_cloud_path(path: T | CloudPath) -> T | Path:
    if not isinstance(path, (CloudPath)):
        return path

    if isinstance(path, LocalPath):
        return path.client._cloud_path_to_local(path)

    # TODO: Handle actual cloud paths here. Will need to also return credentials/storage_options dict
    raise ValueError("Unsupported path type. Expected LocalPath or CloudPath.")
