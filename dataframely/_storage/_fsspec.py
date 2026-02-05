# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from fsspec import AbstractFileSystem


def get_file_prefix(fs: AbstractFileSystem) -> str:
    match fs.protocol:
        case "file":
            return ""
        case str():
            return f"{fs.protocol}://"
        case ["file", *_]:
            return ""
        case [str(proto), *_]:
            return f"{proto}://"
        case _:
            raise ValueError(f"Unexpected fs.protocol: {fs.protocol}")
