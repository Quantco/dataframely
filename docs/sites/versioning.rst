Versioning policy and breaking changes
======================================

Dataframely uses `semantic versioning <https://semver.org/>`_.
This versioning scheme is designed to make it easy for users to anticipate what types of change they can expect from a given version update in their dependencies.
We generally recommend that users take measures to control dependency versions. Personally, we like to use ``pixi`` as a package manager, which comes with builtin
support for lockfiles. Many other package managers support similar functionality. When updating the lockfiles, we recommend to use automated testing
to ensure that user code still works with newer versions of dependencies such as ``dataframely``.

Most importantly, semantic versioning implies that breaking changes of user-facing functionality are only introduced in **major releases**.
We therefore recommend that users are particularly vigilant when updating their environments to a newer major release of `dataframely`.
As always, automated testing is useful here, but we also recommend checking the release notes `published on GitHub <https://github.com/Quantco/dataframely/releases>`_.

In order to give users a heads-up before breaking changes are released, we introduce `FutureWarnings <https://docs.python.org/3/library/exceptions.html#FutureWarning>`_ .
Warnings are the most direct and effective tool at our disposal for reaching users directly.
We therefore generally recommend that users do not silence such warnings explicitly, but instead migrate their code proactively, whenever possible.
However, we also understand that the need for migration may catch users at an inconvenient time, and a temporary band aid solution might be required.
Users can disable ``FutureWarnings`` either through `python builtins <https://docs.python.org/3/library/warnings.html#warnings.filterwarnings>`_,
builtins from tools like `pytest <https://docs.pytest.org/en/stable/how-to/capture-warnings.html#controlling-warnings>`_ ,
or by setting the ``DATAFRAMELY_NO_FUTURE_WARNINGS`` environment variable to ``true`` or ``1``.
