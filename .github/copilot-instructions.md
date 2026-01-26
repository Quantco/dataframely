# Dataframely - Coding Agent Instructions

## Project Overview

Dataframely is a declarative, polars-native data frame validation library. It validates schemas and data content in
polars DataFrames using native polars expressions and a custom Rust-based polars plugin for high performance. It
supports validating individual data frames via `Schema` classes and interconnected data frames via `Collection` classes.

## Tech Stack

### Core Technologies

- **Python**: Primary language for the public API
- **Rust**: Backend for polars plugin and custom regex operations
- **Polars**: Only supported data frame library
- **pyo3 & maturin**: Rust-Python bindings and build system
- **pixi**: Primary environment and task manager (NOT pip/conda directly)

### Build System

- **maturin**: Builds the Rust extension module `dataframely._native`
- **Cargo**: Rust dependency management
- Rust toolchain specified in `rust-toolchain.toml` with clippy and rustfmt components

## Environment Setup

**CRITICAL**: Always use `pixi` commands - never run `pip`, `conda`, `python`, or `cargo` directly unless specifically
required for Rust-only operations.

### Initial Setup

Unless already performed via external setup steps:

```bash
# Install Rust toolchain
rustup show

# Install pixi environment and dependencies
pixi install

# Build and install the package locally (REQUIRED after Rust changes)
pixi run postinstall
```

### After Rust Code Changes

**Always run** `pixi run postinstall` after modifying any Rust code in `src/` to rebuild the native extension.

## Development Workflow

### Running Tests

```bash
# Run all tests (excludes S3 tests by default)
pixi run test

# Run tests with S3 backend (requires moto server)
pixi run test -m s3

# Run specific test file or directory
pixi run test tests/schema/

# Run with coverage
pixi run test-coverage

# Run benchmarks
pixi run test-bench
```

### Code Quality

**NEVER** run linters/formatters directly. Use pre-commit:

```bash
# Run all pre-commit hooks
pixi run pre-commit run
```

Pre-commit handles:

- **Python**: ruff (lint & format), mypy (type checking), docformatter
- **Rust**: cargo fmt, cargo clippy
- **Other**: prettier (md/yml), taplo (toml), license headers, trailing whitespace

### Building Documentation

```bash
# Build documentation
pixi run -e docs postinstall
pixi run docs

# Open in browser (macOS)
open docs/_build/html/index.html
```

## Project Structure

```
dataframely/              # Python package
  schema.py              # Core Schema class for DataFrame validation
  collection/            # Collection class for validating multiple interconnected DataFrames
  columns/               # Column type definitions (String, Integer, Float, etc.)
  testing/               # Testing utilities (factories, masks, storage mocks)
  _storage/              # Storage backends (Parquet, Delta Lake)
  _rule.py               # Rule decorator for validation rules
  _plugin.py             # Polars plugin registration
  _native.pyi            # Type stubs for Rust extension

src/                     # Rust source code
  lib.rs                 # PyO3 module definition
  polars_plugin/         # Custom polars plugin for validation
  regex/                 # Custom regex operations

tests/                   # Unit tests (mirrors dataframely/ structure)
  benches/               # Benchmark tests
  conftest.py            # Shared pytest fixtures (including s3_server)

docs/                    # Sphinx documentation
  guides/                # User guides and examples
  api/                   # Auto-generated API reference
```

## Pixi Environments

Multiple environments for different purposes:

- **default**: Base Python + core dependencies
- **dev**: Includes jupyter for notebooks
- **test**: Testing dependencies (pytest, moto, boto3, etc.)
- **docs**: Documentation building (sphinx, myst-parser, etc.)
- **lint**: Linting and formatting tools
- **optionals**: Optional dependencies (pydantic, deltalake, pyarrow, sqlalchemy)
- **py310-py314**: Python version-specific environments

Use `-e <env>` to run commands in specific environments:

```bash
pixi run -e test test
pixi run -e docs docs
```

## API Design Principles

### Critical Guidelines

1. **NO BREAKING CHANGES**: Public API must remain backward compatible
2. **100% Test Coverage**: All new code requires tests
3. **Documentation Required**: All public features need docstrings + API docs
4. **Cautious API Extension**: Avoid adding to public API unless necessary

### Public API

Public exports are in `dataframely/__init__.py`. Main components:

- **Schema classes**: `Schema` for DataFrame validation
- **Collection classes**: `Collection`, `CollectionMember` for multi-DataFrame validation
- **Column types**: `String`, `Integer`, `Float`, `Bool`, `Date`, `Datetime`, etc.
- **Decorators**: `@rule()`, `@filter()`
- **Type hints**: `DataFrame[Schema]`, `LazyFrame[Schema]`, `Validation`

## Common Pitfalls & Solutions

### S3 Testing

The `s3_server` fixture in `tests/conftest.py` uses `subprocess.Popen` to start moto_server on port 9999. This is a **workaround** for a polars issue with ThreadedMotoServer. When the polars issue is fixed, it should be replaced with ThreadedMotoServer (code is commented in the file).

**Note**: CI skips S3 tests by default. Run with `pixi run test -m s3` when modifying storage backends.

## Testing Strategy

- Tests are organized by module, mirroring the `dataframely/` structure
- Use `dy.Schema.sample()` for generating test data
- Test both eager (`DataFrame`) and lazy (`LazyFrame`) execution
- S3 tests use moto server fixture from `conftest.py`
- Benchmark tests in `tests/benches/` use pytest-benchmark

## Validation Pattern

Typical usage pattern:

```python
class MySchema(dy.Schema):
    col = dy.String(nullable=False)

    @dy.rule()
    def my_rule(cls) -> pl.Expr:
        return pl.col("col").str.len_chars() > 0

# Validate and cast
validated_df: dy.DataFrame[MySchema] = MySchema.validate(df, cast=True)
```

## Key Configuration Files

- `pixi.toml`: Environment and task definitions
- `pyproject.toml`: Python package metadata, tool configurations (ruff, mypy, pytest)
- `Cargo.toml`: Rust dependencies and build settings
- `.pre-commit-config.yaml`: All code quality checks
- `rust-toolchain.toml`: Rust nightly version specification

## When Making Changes

1. **Python code**: Run `pixi run pre-commit run` before committing
2. **Rust code**: Run `pixi run postinstall` to rebuild, then run tests
3. **Tests**: Ensure `pixi run test` passes. If changes might affect storage backends, use `pixi run test -m s3`.
4. **Documentation**: Update docstrings
5. **API changes**: Ensure backward compatibility or document migration path

### Pull request titles (required)

Pull request titles must follow the Conventional Commits format: `<type>[!]: <Subject>`

Allowed `type` values:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to our CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files
- `revert`: Reverts a previous commit

Additional rules:

- Use `!` only for **breaking changes**
- `Subject` must start with an **uppercase** letter and must **not** end with `.` or a trailing space

## Performance Considerations

- Validation uses native polars expressions for performance
- Custom Rust plugin for advanced validation logic
- Lazy evaluation supported via `LazyFrame` for large datasets
- Avoid materializing data unnecessarily in validation rules
