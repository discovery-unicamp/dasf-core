.. _installation:

==========================
Installation Guide
==========================

The installation can be done using `conda` or `docker`.

Using Docker
--------------

To install DASF using docker, you must in the go to the `build/` directory and
execute the command below directory according to your build type: `cpu` or
`gpu`.

.. code-block:: bash

  ./build_docker.sh <cpu|gpu>


The `dasf` image will be created and be ready to use. Once it is ready, you
can start a jupyter instance by executing the command:

.. code-block:: bash

  ./start_jupyter_server.sh


Using Conda
-------------

If you just want to create a base Conda environment for DASF, you need to
create it, using the respective YAML file based on architecture: for CPUs
or GPUs. The environment name is always `dasf`.


.. code-block:: bash

  conda env create -f build/conda/{cpu,gpu}/environment.yml


Development version
--------------------

To install this development version, all you need to do is run `pip` from the
root project directory (the same where `pyproject.toml` lives).

.. code-block:: bash

  python -m pip install .

Testing
--------

If you have a working environment with DASF installed, you can execute the all
the test set. Make sure you have all development packages installed such as
**pytest**, **parameterized** and **mock**. To run, you need to execute
`pytest` from the `tests/` directory.

.. code-block:: bash

  pytest tests/
