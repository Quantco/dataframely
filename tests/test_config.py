# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import dataframely as dy


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
