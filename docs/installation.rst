.. _installation:

Installation
************

| The code is written for Python 3.6+.
| The following packages are needed for a minimum working installation

* `NumPy <https://numpy.org/>`_
* `SciPy <https://scipy.org/>`_

The `package <https://pypi.org/project/eminus/>`_ and all necessary dependencies can be installed with

.. code-block:: bash

   pip install eminus

Alternatively, you can create an installation by downloading the source code

.. code-block:: bash

   git clone https://gitlab.com/nextdft/eminus.git
   cd eminus
   pip install .

To also install all optional dependencies to use built-in addons, use either

.. code-block:: bash

   pip install eminus[addons]

or for an installation after downloading the source code, use

.. code-block:: bash

   pip install .[addons]
