.. _installation:

Installation
************

| The code is written for Python 3.6+.
| The following packages are needed for a working installation

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_

The package and all necessary dependencies can be installed with

.. code-block:: bash

   git clone https://gitlab.com/wangenau/plainedft.git
   cd plainedft
   pip install .

To also install all optional dependecies to use built-in addons, use

.. code-block:: bash


   pip install .[addons]
