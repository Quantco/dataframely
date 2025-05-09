Breaking changes
================

Breaking changes to existing APIs are naturally disruptive to users, but sometimes
necessary in order to introduce new functionality.
When we intend to make breaking changes to user-facing APIs in dataframely, we aim to make it easy for users to anticipate such changes and migrate their code to the new behavior.

``FutureWarnings``
----------------
Wherever possible, we introduce `FutureWarnings <https://docs.python.org/3/library/exceptions.html#FutureWarning>`_ before the breaking changes take effect. Warnings are the most direct and effective tool at our disposal for reaching users directly. We therefore generally recommend that users do not silence such warnings explicitly. We believe that the best way of silencing a warning should be to migrate to new behavior early. However, we also understand that the need for migration may catch users at an inconvenient time, and a temporary band aid solution might be required. Users can disable ``FutureWarnings`` either through `python builtins <https://docs.python.org/3/library/warnings.html#warnings.filterwarnings>`_, builtins from tools `like pytest <https://docs.pytest.org/en/stable/how-to/capture-warnings.html#controlling-warnings>`_ , or by setting the ``DATAFRAMELY_NO_FUTURE_WARNINGS`` environment variable to ``true`` or ``1``.
