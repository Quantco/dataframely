# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import math
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

import dataframely as dy
from dataframely.random import Generator
from dataframely.testing import create_schema


@pytest.fixture()
def generator() -> Generator:
    return Generator(seed=42)


# -------------------------------- GENERAL PROPERTIES -------------------------------- #


def test_seeding_constant() -> None:
    results = {Generator(seed=42).sample_seed() for _ in range(1000)}
    assert len(results) == 1


def test_seeding_nonconstant() -> None:
    results = {Generator().sample_seed() for _ in range(1000)}
    assert len(results) > 1


@pytest.mark.parametrize(
    "fn",
    [
        lambda generator, n: generator.sample_int(n, min=0, max=5),
        lambda generator, n: generator.sample_bool(n),
        lambda generator, n: generator.sample_float(n, min=0, max=5),
        lambda generator, n: generator.sample_string(n, regex="[abc]"),
        lambda generator, n: generator.sample_choice(n, choices=[1, 2, 3]),
        lambda generator, n: generator.sample_time(n, min=dt.time(0, 0), max=None),
        lambda generator, n: generator.sample_date(
            n, min=dt.date(1970, 1, 1), max=None
        ),
        lambda generator, n: generator.sample_datetime(
            n, min=dt.datetime(1970, 1, 1), max=None
        ),
        lambda generator, n: generator.sample_duration(
            n, min=dt.timedelta(), max=dt.timedelta(days=1)
        ),
    ],
)
@pytest.mark.parametrize("n", [1, 100])
def test_sample_correct_n(
    generator: Generator, fn: Callable[[Generator, int], pl.Series], n: int
) -> None:
    assert len(fn(generator, n)) == n


@pytest.mark.parametrize(
    "fn",
    [
        lambda generator, n, prob: generator.sample_int(
            n, min=0, max=5, null_probability=prob
        ),
        lambda generator, n, prob: generator.sample_bool(n, null_probability=prob),
        lambda generator, n, prob: generator.sample_float(
            n, min=0, max=5, null_probability=prob
        ),
        lambda generator, n, prob: generator.sample_string(
            n, regex="[abc]", null_probability=prob
        ),
        lambda generator, n, prob: generator.sample_choice(
            n, choices=[1, 2, 3], null_probability=prob
        ),
        lambda generator, n, prob: generator.sample_time(
            n, min=dt.time(0, 0), max=None, null_probability=prob
        ),
        lambda generator, n, prob: generator.sample_date(
            n, min=dt.date(1970, 1, 1), max=None, null_probability=prob
        ),
        lambda generator, n, prob: generator.sample_datetime(
            n, min=dt.datetime(1970, 1, 1), max=None, null_probability=prob
        ),
        lambda generator, n, prob: generator.sample_duration(
            n, min=dt.timedelta(), max=dt.timedelta(days=1), null_probability=prob
        ),
    ],
)
@pytest.mark.parametrize("null_probability", [0, 0.1])
def test_sample_correct_null_probability(
    generator: Generator,
    fn: Callable[[Generator, int, float], pl.Series],
    null_probability: float,
) -> None:
    n = 100_000
    assert math.isclose(
        fn(generator, n, null_probability).is_null().sum() / n,
        null_probability,
        abs_tol=0 if null_probability == 0 else 0.01,
    )


# ---------------------------- INDIVIDUAL SAMPLING METHODS --------------------------- #


def test_sample_int(generator: Generator) -> None:
    samples = generator.sample_int(100_000, min=1, max=4)
    assert samples.min() == 1
    assert samples.max() == 3
    assert math.isclose(samples.mean(), 2, abs_tol=0.01)  # type: ignore


@pytest.mark.parametrize("p_true", [0, 0.1, 0.5, None, 0.9, 1.0])
def test_sample_bool(generator: Generator, p_true: bool | None) -> None:
    samples = generator.sample_bool(100_000, p_true=p_true)
    assert math.isclose(samples.mean(), p_true or 0.5, abs_tol=0.01)  # type: ignore


def test_sample_float(generator: Generator) -> None:
    samples = generator.sample_float(100_000, min=1, max=3)
    assert samples.min() >= 1  # type: ignore
    assert samples.max() < 3  # type: ignore
    assert math.isclose(samples.mean(), 2, abs_tol=0.01)  # type: ignore


def test_sample_string(generator: Generator) -> None:
    samples = generator.sample_string(100_000, regex="[abc]d")
    assert (samples.str.len_bytes() == 2).all()


def test_sample_choice(generator: Generator) -> None:
    samples = generator.sample_choice(100_000, choices=[1, 2, 3])
    assert np.allclose(
        samples.value_counts().sort("").get_column("count") / 100_000,
        [1 / 3, 1 / 3, 1 / 3],
        atol=0.01,
    )


@pytest.mark.parametrize("weight_factor", [0.01, 1, 1000])
def test_sample_choice_weights(generator: Generator, weight_factor: float) -> None:
    with pytest.raises(ValueError):
        generator.sample_choice(
            100, choices=[1, 2, 3], null_probability=0.1, weights=[1]
        )

    samples = generator.sample_choice(
        100_000,
        choices=[0, 1],
        weights=[0.2 * weight_factor, 0.8 * weight_factor],
    )
    assert np.allclose(
        samples.value_counts().sort("").get_column("count") / 100_000,
        [0.2, 0.8],
        atol=0.01,
    )


@pytest.mark.parametrize(
    ("fn", "column_type", "resolution"),
    [
        (
            lambda generator, resolution: generator.sample_time(
                100_000, min=dt.time(12, 0), max=None, resolution=resolution
            ),
            dy.Time,
            "1h",
        ),
        (
            lambda generator, resolution: generator.sample_datetime(
                100_000, min=dt.datetime(1970, 1, 1), max=None, resolution=resolution
            ),
            dy.Datetime,
            "12h",
        ),
        (
            lambda generator, resolution: generator.sample_date(
                100_000, min=dt.date(1970, 1, 1), max=None, resolution=resolution
            ),
            dy.Date,
            "1y",
        ),
        (
            lambda generator, resolution: generator.sample_duration(
                100_000,
                min=dt.timedelta(minutes=30),
                max=dt.timedelta(minutes=60),
                resolution=resolution,
            ),
            dy.Duration,
            "5m",
        ),
    ],
)
def test_sample_resolutions(
    generator: Generator,
    fn: Callable[[Generator, str], pl.Series],
    column_type: type[dy.Column],
    resolution: str,
) -> None:
    samples = fn(generator, resolution)
    schema = create_schema("test", {"a": column_type(resolution=resolution)})  # type: ignore
    schema.validate(samples.to_frame("a"))


# ---------------------------------- ARG VIOLATIONS ---------------------------------- #


@pytest.mark.parametrize(
    "fn",
    [
        lambda generator: generator.sample_time(
            1, min=dt.time(0, 15), max=None, resolution="1h"
        ),
        lambda generator: generator.sample_time(
            1, min=dt.time(0, 0), max=dt.time(0, 15), resolution="1h"
        ),
        lambda generator: generator.sample_date(
            1, min=dt.date(1970, 1, 5), max=None, resolution="1mo"
        ),
        lambda generator: generator.sample_date(
            1, min=dt.date(1970, 1, 1), max=dt.date(1970, 1, 5), resolution="1mo"
        ),
        lambda generator: generator.sample_datetime(
            1, min=dt.datetime(1970, 1, 1, 12), max=None, resolution="1d"
        ),
        lambda generator: generator.sample_datetime(
            1,
            min=dt.datetime(1970, 1, 1),
            max=dt.datetime(1970, 1, 1, 12),
            resolution="1d",
        ),
        lambda generator: generator.sample_duration(
            1,
            min=dt.timedelta(seconds=90),
            max=dt.timedelta(seconds=120),
            resolution="1m",
        ),
        lambda generator: generator.sample_duration(
            1,
            min=dt.timedelta(seconds=60),
            max=dt.timedelta(seconds=90),
            resolution="1m",
        ),
    ],
)
def test_sample_invalid_arg(
    generator: Generator, fn: Callable[[Generator], pl.Series]
) -> None:
    with pytest.raises(ValueError):
        fn(generator)
