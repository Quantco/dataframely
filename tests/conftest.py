# Copyright (c) QuantCo 2025-2025
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
def any_tmp_path(request: pytest.FixtureRequest) -> str:
    return str(request.getfixturevalue(request.param))
