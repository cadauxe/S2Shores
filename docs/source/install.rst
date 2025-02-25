.. _install:

======================
Installation
======================

To install **S2shores**, create a conda environment and install the correct version of *libgdal*. Then, install *s2shores* with *pip*.

.. code-block:: console

    $ conda create -n env_name -y
    $ conda activate env_name
    $ conda install python=3.12 libgdal=3.8 -c conda-forge -y
    $ pip install s2shores --no-cache-dir
