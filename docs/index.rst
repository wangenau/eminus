..
   SPDX-FileCopyrightText: 2021 The eminus developers
   SPDX-License-Identifier: Apache-2.0

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
   examples/examples.rst
   modules.rst
   citation.rst
   changelog.rst
   license.rst
   nomenclature.rst
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
      :link: examples/examples
      :link-type: doc
      :columns: 6

   .. grid-item-card:: :octicon:`tools` Source code
      :link: https://gitlab.com/wangenau/eminus
      :link-type: url
      :columns: 6

.. image:: https://img.shields.io/pypi/v/eminus?color=1a962b&logo=python&logoColor=a0dba2&label=Version
   :target: https://pypi.org/project/eminus

.. image:: https://img.shields.io/badge/license-Apache2.0-1a962b?logo=python&logoColor=a0dba2&label=License
   :target: https://wangenau.gitlab.io/eminus/license.html

.. image:: https://img.shields.io/pypi/pyversions/eminus?color=1a962b&logo=python&logoColor=a0dba2&label=Python
   :target: https://wangenau.gitlab.io/eminus/installation.html

.. image:: https://img.shields.io/gitlab/pipeline-coverage/wangenau%2Feminus?branch=main&color=1a962b&logo=gitlab&logoColor=a0dba2&label=Coverage
   :target: https://wangenau.gitlab.io/eminus/htmlcov

.. image:: https://img.shields.io/badge/Chat-Discord-1a962b?logo=discord&logoColor=a0dba2
   :target: https://discord.gg/k2XwdMtVec

.. image:: https://img.shields.io/badge/DOI-10.1016/j.softx.2025.102035-1a962b?logo=DOI&logoColor=a0dba2
   :target: https://doi.org/10.1016/j.softx.2025.102035

eminus is a pythonic electronic structure theory code.
It implements plane wave density functional theory (DFT) with self-interaction correction (SIC) functionalities.
The goal is to create a simple code that is easy to read and easy to extend while using minimal dependencies.
It is built upon the `DFT++ <https://arxiv.org/abs/cond-mat/9909130>`_ pragmas proposed by Tomas Arias et al. that aim to let programming languages and theory coincide.
This can be shown by, e.g., solving the Poisson equation. In the operator notation of DFT++ the equation reads

.. math::

   \phi(\boldsymbol r) = -4\pi\mathcal L^{-1}\mathcal O\mathcal J n(\boldsymbol r).

The corresponding Python code (implying that the operators have been implemented properly) reads

.. code-block:: python

   def get_phi(atoms, n):
      return -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))
