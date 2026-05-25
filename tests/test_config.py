# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import dataframely as dy


def test_config_global() -> None:
    dy.Config.set_max_sampling_iterations(50)
    assert dy.Config.options["max_sampling_iterations"] == 50
    dy.Config.restore_defaults()


def test_config_global_max_failure_examples() -> None:
    try:
        dy.Config.set_max_failure_examples(7)
        assert dy.Config.options["max_failure_examples"] == 7
    finally:
        dy.Config.restore_defaults()
    assert dy.Config.options["max_failure_examples"] == 0


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
