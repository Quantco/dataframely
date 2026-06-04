# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import uuid
from collections.abc import Iterator

import boto3
import pytest


@pytest.fixture(scope="session")
def s3_server() -> Iterator[str]:
    # FIXME: Replace with the commented-out code below once
    #  https://github.com/pola-rs/polars/pull/24922 is released.
    process = subprocess.Popen(["moto_server", "--port", "9999"])
    yield "http://localhost:9999"
    process.terminate()
    process.wait()

    # server = ThreadedMotoServer(port=0)
    # server.start()
    # host, port = server.get_host_and_port()
    # yield f"http://{host}:{port}"
    # server.stop()


@pytest.fixture(scope="session")
def s3_bucket(s3_server: str) -> str:
    client = boto3.client(
        "s3", endpoint_url=s3_server, aws_access_key_id="", aws_secret_access_key=""
    )
    client.create_bucket(Bucket="test")
    return "s3://test"


@pytest.fixture()
def s3_tmp_path(s3_server: str, s3_bucket: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("AWS_ENDPOINT_URL", s3_server)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_ALLOW_HTTP", "true")
    monkeypatch.setenv("AWS_S3_ALLOW_UNSAFE_RENAME", "true")
    return f"{s3_bucket}/{str(uuid.uuid4())}"


@pytest.fixture()
def s3_isolated(
    s3_server: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, dict[str, str]]:
    """A freshly-named bucket that is only reachable via the returned ``storage_options``.

    Polars caches object stores per bucket, and these caches live Rust-side and are not
    cleared by ``monkeypatch.delenv``. A bucket configured once from ``AWS_*`` env vars
    (e.g. by :func:`s3_tmp_path`) therefore stays reachable without ``storage_options``,
    which would let a read silently succeed even if ``storage_options`` was dropped. A
    unique bucket has no such cached store, so reaching it requires forwarding
    ``storage_options`` to every read.
    """
    for var in (
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ALLOW_HTTP",
        "AWS_S3_ALLOW_UNSAFE_RENAME",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
    ):
        monkeypatch.delenv(var, raising=False)
    bucket = f"isolated-{uuid.uuid4()}"
    boto3.client(
        "s3", endpoint_url=s3_server, aws_access_key_id="", aws_secret_access_key=""
    ).create_bucket(Bucket=bucket)
    return f"s3://{bucket}", {
        "aws_access_key_id": "testing",
        "aws_secret_access_key": "testing",
        "aws_endpoint_url": s3_server,
        "aws_region": "us-east-1",
        "aws_allow_http": "true",
    }


@pytest.fixture()
def any_tmp_path(request: pytest.FixtureRequest) -> str:
    return str(request.getfixturevalue(request.param))
