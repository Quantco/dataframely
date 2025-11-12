---
applyTo: tests/**/*.py
---

# Testing Guidelines

- Instead of writing new tests, existing tests should ideally be parametrized using `@pytest.mark.parametrize`
  unless the parametrization is impractical, e.g. by adding more parameters
- Tests should not use docstrings unless they are _very_ complex
