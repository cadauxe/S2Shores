====================
S2Shores
====================

This project gathers several estimators to be able to compute bathymetry from standard format such as sentinel2 or geotiff.

It uses methods based on the inversion of wave peaks and data provider services (delta time b.w. frames, gravity depending on latitude, distance to shore) for bathymetry estimation.

S2shores can be used through its Python API or Command Line Interface.

Content
==================

* .. toctree::
   :maxdepth: 1
   :caption: Installation

   Installation procedure <install>


* .. toctree::
   :caption: Tutorials
   :maxdepth: 1

   Tutorial <tutorial>
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
