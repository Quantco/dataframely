# Development

Thanks for deciding to work on `dataframely`!
You can create a development environment with the following steps:

## Install Tooling

To work on dataframely, you'll need to install

- [`pixi`](https://pixi.sh/latest/) to manage the Python environment
- [`rustup`](https://rustup.rs/) to manage the Rust toolchain for compiling dataframely

## Environment Installation

```bash
git clone https://github.com/Quantco/dataframely
cd dataframely
rustup show
pixi install
```

Next make sure to install the package locally and set up pre-commit hooks:

```bash
pixi run postinstall
pixi run pre-commit-install
```

## Running the tests

```bash
pixi run test
```

You can adjust the `tests/` path to run tests in a specific directory or module.

## Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) together
with [MyST](https://myst-parser.readthedocs.io/), and write user documentation in markdown.
If you are not yet familiar with this setup,
the [MyST docs for Sphinx](https://myst-parser.readthedocs.io/en/v0.17.2/sphinx/intro.html) are a good starting point.

When updating the documentation, you can compile a localized build of the
documentation and then open it in your web browser using the commands below:

```bash
# Run build
pixi run -e docs postinstall
pixi run docs

# Open documentation
open docs/_build/html/index.html
```
