:html_theme.sidebar_secondary.remove: true

.. role:: hidden

{{ name | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ name }}
    :members:
    :exclude-members: as_dict, from_dict, matches, pyarrow_field, pyarrow_dtype, sqlalchemy_dtype, sqlalchemy_column, validate_dtype, validation_rules
    :autosummary:
    :autosummary-nosignatures:
