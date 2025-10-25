===
I/O
===

Writing Data
------------

.. currentmodule:: dataframely
.. autosummary::
    :toctree: _gen/

    Schema.write_parquet
    Schema.sink_parquet
    Schema.write_delta

Reading Data
------------

.. currentmodule:: dataframely
.. autosummary::
    :toctree: _gen/

    Schema.read_parquet
    Schema.scan_parquet
    Schema.read_delta
    Schema.scan_delta

Schema Serialization
--------------------

.. currentmodule:: dataframely
.. autosummary::
    :toctree: _gen/

    Schema.serialize
    deserialize_schema
    read_parquet_metadata_schema
