===
I/O
===

Writing Data
------------

.. currentmodule:: dataframely
.. autosummary::
    :toctree: _gen/

    Collection.write_parquet
    Collection.sink_parquet
    Collection.write_delta

Reading Data
------------

.. currentmodule:: dataframely
.. autosummary::
    :toctree: _gen/

    Collection.read_parquet
    Collection.scan_parquet
    Collection.read_delta
    Collection.scan_delta

Collection Serialization
------------------------

.. currentmodule:: dataframely
.. autosummary::
    :toctree: _gen/

    Collection.serialize
    deserialize_collection
    read_parquet_metadata_collection
