# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import collections
import os
import uuid
from collections.abc import Iterator
from typing import cast

import pytest
from minio import Minio
from minio.deleteobjects import DeleteObject


def _delete_prefix(client: Minio, bucket: str, prefix: str) -> None:
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    remove_objects = [DeleteObject(cast(str, obj.object_name)) for obj in objects]
    collections.deque(
        client.remove_objects(bucket, remove_objects),
        maxlen=0,
    )


@pytest.fixture(scope="session")
def s3_endpoint() -> str:
    return os.environ["AWS_ENDPOINT_URL"].split("://")[-1]


@pytest.fixture(scope="session")
def s3_client(s3_endpoint: str) -> Minio:
    return Minio(
        endpoint=s3_endpoint,
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        secure=False,
    )


@pytest.fixture(scope="session")
def s3_bucket(s3_client: Minio) -> Iterator[str]:
    bucket_name = "test"

    if s3_client.bucket_exists(bucket_name):
        _delete_prefix(s3_client, bucket_name, "")
        s3_client.remove_bucket(bucket_name)

    s3_client.make_bucket(bucket_name)
    yield bucket_name
    _delete_prefix(s3_client, bucket_name, "")
    s3_client.remove_bucket(bucket_name)


@pytest.fixture()
def s3_tmp_path(s3_client: Minio, s3_bucket: str) -> Iterator[str]:
    prefix = str(uuid.uuid4())
    path = os.path.join(f"s3://{s3_bucket}", prefix)
    yield path
    _delete_prefix(s3_client, s3_bucket, prefix)
