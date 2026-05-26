# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest

import dataframely as dy
from dataframely import config as _config


def test_config_global() -> None:
    dy.Config.set_max_sampling_iterations(50)
    assert dy.Config.options["max_sampling_iterations"] == 50
    dy.Config.restore_defaults()


def test_config_local() -> None:
    try:
        with dy.Config(max_sampling_iterations=35):
            assert dy.Config.options["max_sampling_iterations"] == 35
        assert dy.Config.options["max_sampling_iterations"] == 10_000
    finally:
        dy.Config.restore_defaults()


def test_config_local_nested() -> None:
    try:
        with dy.Config(max_sampling_iterations=35):
            assert dy.Config.options["max_sampling_iterations"] == 35
            with dy.Config(max_sampling_iterations=20):
                assert dy.Config.options["max_sampling_iterations"] == 20
            assert dy.Config.options["max_sampling_iterations"] == 35
        assert dy.Config.options["max_sampling_iterations"] == 10_000
    finally:
        dy.Config.restore_defaults()


def test_config_global_local() -> None:
    try:
        dy.Config.set_max_sampling_iterations(50)
        assert dy.Config.options["max_sampling_iterations"] == 50
        with dy.Config(max_sampling_iterations=35):
            assert dy.Config.options["max_sampling_iterations"] == 35
        assert dy.Config.options["max_sampling_iterations"] == 50
    finally:
        dy.Config.restore_defaults()


def test_config_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFRAMELY_MAX_SAMPLING_ITERATIONS", "123")
    try:
        importlib.reload(_config)
        assert _config.Config.options["max_sampling_iterations"] == 123
    finally:
        monkeypatch.delenv("DATAFRAMELY_MAX_SAMPLING_ITERATIONS")
        importlib.reload(_config)
        # Re-bind dy.Config to the reloaded module's class to keep state consistent.
        dy.Config = _config.Config  # type: ignore
        dy.Config.restore_defaults()


def test_config_env_var_not_reread_after_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAFRAMELY_MAX_SAMPLING_ITERATIONS", "777")
    assert dy.Config.options["max_sampling_iterations"] == 10_000
