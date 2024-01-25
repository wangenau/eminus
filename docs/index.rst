.. _index:

Home
****

.. meta::
   :description: Documentation website for the eminus code.
   :author: Wanja Timm Schulze

.. toctree::
   :caption: Contents
   :maxdepth: 1
   :numbered:
   :hidden:

   self
   overview.rst
   installation.rst
   theory.rst
   _examples/examples.rst
   nomenclature.rst
   modules.rst
   changelog.rst
   citation.rst
   license.rst
   further.rst

.. image:: /_static/logo/eminus.svg
   :class: only-light
   :align: center
   :width: 65%

.. image:: /_static/logo/eminus_dark.svg
   :class: only-dark
   :align: center
   :width: 65%

|

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :octicon:`rocket` Installation
      :link: installation
      :link-type: doc
      :columns: 6

   .. grid-item-card:: :octicon:`code` API reference
      :link: modules
      :link-type: doc
      :columns: 6

   .. grid-item-card:: :octicon:`book` User guide
      :link: _examples/examples
      :link-type: doc
      :columns: 6

   .. grid-item-card:: :octicon:`tools` Source code
      :link: https://gitlab.com/wangenau/eminus
      :link-type: url
      :columns: 6

.. image:: https://img.shields.io/pypi/v/eminus?color=1a962b
   :target: https://pypi.org/project/eminus

.. image:: https://gitlab.com/wangenau/eminus/badges/main/coverage.svg
   :target: https://wangenau.gitlab.io/eminus/htmlcov

.. image:: https://img.shields.io/pypi/pyversions/eminus?color=green
   :target: https://wangenau.gitlab.io/eminus/installation.html

.. image:: https://img.shields.io/badge/license-Apache2.0-yellowgreen
   :target: https://wangenau.gitlab.io/eminus/license.html

.. image:: https://zenodo.org/badge/431079841.svg
   :target: https://zenodo.org/badge/latestdoi/431079841

eminus is a pythonic plane wave density functional theory (DFT) code with self-interaction correction (SIC) functionalities.
The goal is to create a simple code that is easy to read and easy to extend while using minimal dependencies.
It is built upon the `DFT++ <https://arxiv.org/abs/cond-mat/9909130>`_ pragmas proposed by Tomas Arias et al. that aim to let programming languages and theory coincide.
This can be shown by, e.g., solving the Poisson equation. In the operator notation of DFT++ the equation reads

.. math::

   \boldsymbol \phi = 4\pi\hat L^{-1}\hat O\hat J \boldsymbol n.

The corresponding Python code (implying that the operators have been implemented properly) reads

.. code-block:: python

   phi = -4 * np.pi * Linv(O(J(n)))
