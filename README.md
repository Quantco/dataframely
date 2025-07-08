<!-- LOGO -->
<br />

<div align="center">

  <h3 align="center">
  <code>dataframely</code> — A declarative, 🐻‍❄️-native data frame validation library
  </h3>

[![CI](https://img.shields.io/github/actions/workflow/status/quantco/dataframely/ci.yml?style=flat-square&branch=main)](https://github.com/quantco/dataframely/actions/workflows/ci.yml)
[![Nightly CI](https://img.shields.io/github/actions/workflow/status/quantco/dataframely/nightly.yml?style=flat-square&branch=main)](https://github.com/quantco/dataframely/actions/workflows/nightly.yml)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/dataframely?logoColor=white&logo=conda-forge&style=flat-square)](https://prefix.dev/channels/conda-forge/packages/dataframely)
[![pypi-version](https://img.shields.io/pypi/v/dataframely.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/dataframely)
[![python-version](https://img.shields.io/pypi/pyversions/dataframely?logoColor=white&logo=python&style=flat-square)](https://pypi.org/project/dataframely)
[![codecov](https://codecov.io/gh/Quantco/dataframely/graph/badge.svg?token=QOvhS7Zri2)](https://codecov.io/gh/Quantco/dataframely)

</div>

## 🗂 Table of Contents

- [Introduction](#-introduction)
- [Installation](#-installation)
- [Usage](#-usage)

## 📖 Introduction

Dataframely is a Python package to validate the schema and content of [`polars`](https://pola.rs/) data frames. Its
purpose is to make data pipelines more robust by ensuring that data meets expectations and more readable by adding
schema information to data frame type hints.

## 💿 Installation

You can install `dataframely` using your favorite package manager, e.g., `pixi` or `pip`:

```bash
pixi add dataframely
pip install dataframely
```

## 🎯 Usage

### Defining a data frame schema

```python
import dataframely as dy
import polars as pl

class HouseSchema(dy.Schema):
    zip_code = dy.String(nullable=False, min_length=3)
    num_bedrooms = dy.UInt8(nullable=False)
    num_bathrooms = dy.UInt8(nullable=False)
    price = dy.Float64(nullable=False)

    @dy.rule()
    def reasonable_bathroom_to_bedroom_ratio() -> pl.Expr:
        ratio = pl.col("num_bathrooms") / pl.col("num_bedrooms")
        return (ratio >= 1 / 3) & (ratio <= 3)

    @dy.rule(group_by=["zip_code"])
    def minimum_zip_code_count() -> pl.Expr:
        return pl.len() >= 2
```

### Validating data against schema

```python

import polars as pl

df = pl.DataFrame({
    "zip_code": ["01234", "01234", "1", "213", "123", "213"],
    "num_bedrooms": [2, 2, 1, None, None, 2],
    "num_bathrooms": [1, 2, 1, 1, 0, 8],
    "price": [100_000, 110_000, 50_000, 80_000, 60_000, 160_000]
})

# Validate the data and cast columns to expected types
validated_df: dy.DataFrame[HouseSchema] = HouseSchema.validate(df, cast=True)
```

See more advanced usage examples in the [documentation](https://dataframely.readthedocs.io/en/latest/).
