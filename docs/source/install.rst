.. _install:

======================
Installation
======================

Installation on Linux
=====================

-------------
With Anaconda
-------------

To install S2shores on Linux using Anaconda, you'll first need to create a Conda environment with a compatible version of Python and libgdal.
Once the environment is set up, you can install S2shores via pip.

Please follow these steps:

.. code-block:: console

    $ conda create -n env_name -y
    $ conda activate env_name
    $ conda install gdal=3.9 -c conda-forge -y
    $ pip install s2shores==1.0.0 --no-cache-dir

..
    ----------------
    Without Anaconda
    ----------------

    (To be tested)
    small paragraph describing the install
    Test the install


Installation on Windows
=======================

-------------
With Anaconda
-------------

To install S2shores on Windows using Anaconda, create a Conda environment, install the required version of GDAL, and then use pip to install S2shores.

Please follow these steps:

.. code-block:: console

    $ conda create -n env_name -y
    $ conda activate env_name
    $ conda install gdal=3.9 -c conda-forge -y
    $ pip install s2shores==1.0.0 --no-cache-dir

----------------------------------
Without Anaconda (not recommended)
----------------------------------

Installing S2shores without Anaconda is not recommended, particularly because it requires the use of an unknown wheel file that may be unmaintained or unreliable.

However, if you still prefer to install without Anaconda, make sure that Python is added to your system's PATH.
You’ll need to install GDAL manually from a .whl (Windows Wheel) file, available `here <https://github.com/cgohlke/geospatial-wheels/releases/>`_.

The version you will need is:

    - GDAL-3.9.2-cp312-cp312-win_amd64.whl (for Python 3.12 on 64-bit Windows)

Once you have the appropriate .whl file, please follow these steps:

.. code-block:: console

    $ python -m venv env_name
    $ env_name\Scripts\activate
    $ pip install path_to_the_wheel\GDAL-3.9.2-cp312-cp312-win_amd64.whl
    $ pip install s2shores==win_v1.0.0 --no-cache-dir

