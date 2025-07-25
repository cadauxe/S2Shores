====================
S2Shores
====================

Welcome to S2Shores, a Python package and framework to derive nearshore coastal bathymetries.

S2Shores is a Python package designed to estimate wave characteristics for deriving bathymetries, initially tailored for optical spaceborne data but also compatible with other sensors like RADAR or LiDAR. The general S2Shores' philosophy is to detect bulk wave displacement over a time delay and estimate two of the five key variables (c, T, L, w, k) to solve for bathymetry using wave dispersion relationships (currently only the linear dispersion for free surface waves is implemented). The S2Shores framework contains three methods: spatial DFT, spatial correlation, and temporal correlation. The spatial DFT method, designed for Sentinel 2 imagery, uses two images with a small time delay to find wave directions and characteristics through a Radon Transform and FFT. The spatial correlation method is similar but starts with a 2D image correlation. The temporal correlation method, robust to wave breaking conditions, uses random point sampling and pair-wise time-series correlation, applicable to various spatially distributed time-series data.

- **Spatial DFT**: Uses two images/frames with a small time delay to find multiple wave directions and characteristics through a Radon Transform, FFT, and cross-spectral correlation.
- **Spatial Correlation**: Similar to spatial DFT but starts with a 2D image correlation and applies a Radon Transform.
- **Temporal Correlation**: A time-series correlation method robust to wave breaking conditions, applicable to various types of spatially distributed time-series.

S2Shores is coded in an object-based fashion, with classes separated by specific tasks or specialisms to enable efficient large-scale computing. A global orchestrator ("global_bathymetry") drives local bathymetry estimation using specific methods within the "local_estimator" class. The code modularly handles image processing, wave detection, and wave physics. To modify a method, start with the "local_estimator" class, create a separate branch using git, and commit changes after successful testing. Automatic non-regression tests ensure the quality and precision of S2Shores, with contributions accepted only after passing these tests. The package philosophy emphasizes clean, minimally repetitive code with minimal dependencies, preferring low-level packages like GDAL over umbrella packages like Rasterio. The output is a GIS-ready NetCDF file, easily importable and visualizable in QGIS or xarray in a Python environment.

As a matter of workflow, we recommend reviewing input images to confirm observable wave displacement, as satellite imagery and wave-signal visibility depend on various factors, such as sun angle and season. Lower the output resolution and activate debug mode with plotting to understand S2Shores' process step-by-step for a few points initially and/or have a look at the provided notebooks per method in the /notebooks section. The documentation is a work in progress, and contributions to both code and documentation are encouraged.

The work of S2Shores is an effort of a small group of people, we welcome you to actively contribute to this package to commonly move forward. This is why the chosen license is apache 2.0, which allow you to use the package without contamination in your workflow or usage. The pages, readthedocs for example, are not (yet) perfect or exhaustive, it is merely a work in progress. Slowly but surely we will update the pages, and we invite you when you contribute to the code, to also write a section on the contribution in the manual and read-the-docs. Notebooks are added to provide a clear entry into the code, and perform point analysis per method. When contributing, a small notebook, explaining a modification or new method, would be extremely appreciated. 

Ok, that’s it for the introduction. Enjoy and have fun!



Content
==================

* .. toctree::
   :maxdepth: 1
   :caption: Installation

   Installation procedure <install>


* .. toctree::
   :caption: Tutorials
   :maxdepth: 2

   Tutorial <tutorials/index>
   Contribute to S2Shores <contributing>



* .. toctree::
   :caption: Documentation
   :maxdepth: 1

   Command Line Interface <cli>
   Python API <api>
   More details about S2shores functions <api/modules>


* .. toctree::
   :caption: References
   :maxdepth: 1

   Bibliography <bibliography>
   License <license>
   Authors <authors>
   Changelog <changelog>


.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
