.. _installation:

Installation
************

| The code is written for Python 3.6+.
| The following packages are needed for a minimum working installation

* `NumPy <https://numpy.org>`_
* `SciPy <https://scipy.org>`_

The `package <https://pypi.org/project/eminus>`_ and all necessary dependencies can be installed with

.. code-block:: console

   pip install eminus

Alternatively, you can create an installation by downloading the source code

.. code-block:: console

   git clone https://gitlab.com/wangenau/eminus.git
   cd eminus
   pip install .

To also install all optional dependencies to use built-in extras, use either

.. code-block:: console

   pip install eminus[all]

or for an installation after downloading the source code, use

.. code-block:: console

   pip install .[all]

To install only selected extras, follow the instructions given in :mod:`~eminus.extras`.

All packages have `OSI-approved <https://opensource.org/licenses/alphabetical>`_ licenses and are publicly visible.
