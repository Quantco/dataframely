# Development

Thanks for deciding to work on `dataframely`!
You can create a development environment with the following steps:

## Environment Installation

```bash
git clone https://github.com/Quantco/dataframely
cd dataframely
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

## Building the Documentation

When updating the documentation, you can compile a localized build of the
documentation and then open it in your web browser using the commands below:

```bash
# Run build
pixi run -e docs postinstall
pixi run docs

# Open documentation
open docs/_build/html/index.html
```
