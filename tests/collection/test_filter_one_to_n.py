# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import polars as pl

import dataframely as dy


class CarSchema(dy.Schema):
    vin = dy.String(primary_key=True)
    manufacturer = dy.String(nullable=False)


class CarPartSchema(dy.Schema):
    vin = dy.String(primary_key=True)
    part = dy.String(primary_key=True)
    price = dy.Float64(primary_key=True)


class CarFleet(dy.Collection):
    cars: dy.LazyFrame[CarSchema]
    car_parts: dy.LazyFrame[CarPartSchema]

    @dy.filter()
    def not_car_with_vin_123(self) -> pl.LazyFrame:
        return self.cars.filter(pl.col("vin") != pl.lit("123"))


def test_valid_failure_infos() -> None:
    cars = {"vin": ["123", "456"], "manufacturer": ["BMW", "Mercedes"]}
    car_parts: dict[str, list[Any]] = {
        "vin": ["123", "123", "456"],
        "part": ["Motor", "Wheel", "Motor"],
        "price": [1000, 100, 1000],
    }
    car_fleet, failures = CarFleet.filter(
        {"cars": pl.DataFrame(cars), "car_parts": pl.DataFrame(car_parts)},
        cast=True,
    )

    assert len(car_fleet.cars.collect()) + len(failures["cars"].invalid()) == len(
        cars["vin"]
    )
    assert len(car_fleet.car_parts.collect()) + len(
        failures["car_parts"].invalid()
    ) == len(car_parts["vin"])
