# Dataframely

## Package Management

This repository uses the Pixi package manager. When editing `pixi.toml`, run `pixi lock` afterwards.

When running any commands (like `pytest`), prepend them with `pixi run`.

## Code Style

### Documentation

- Document all public functions/methods and classes using docstrings
  - For functions & methods, use Google Docstrings and include `Args` (if there are any arguments) and `Returns` (if
    there is a return type).
  - Do not include type hints in the docstrings
  - Do not mention default values in the docstrings
- Do not write docstrings for private functions/methods unless the function is highly complex

### License Headers

Do not manually adjust or add license headers. A pre-commit hook will take care of this.

## Testing

- Never use classes for pytest, but only free functions
- Do not put `__init__.py` files into test directories
- Tests should not have docstrings unless they are very complicated or very specific, i.e. warrant a description beyond
  the test's name
- All tests should follow the arrange-act-assert pattern. The respective logical blocks should be distinguished via
  code comments as follows:

  ```python
  def test_method() -> None:
      # Arrange
      ...

      # Act
      ...

      # Assert
      ...
  ```

- If two or more tests are structurally equivalent, they should be merged into a single test and parametrized with
  `@pytest.mark.parametrize`
- If at least two tests share the same logic in the "arrange" step, the respective logic should be extracted into a
  fixture

## Reviewing

When reviewing code changes, make sure that the `SKILL.md` is up-to-date and in line with the public API of this
package.
