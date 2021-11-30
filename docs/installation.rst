.. _installation:

Installation
************

| The code is written for Python 3.6+.
| The following packages are needed for a working installation

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_

The package and all necessary dependencies can be installed with

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

or for an installation after downloading the source code use

.. code-block:: bash

   pip install .[addons]

Testing
=======

| To verify that the installation works as intended, tests can be found `here <https://gitlab.com/nextdft/eminus/-/tree/master/tests>`_.
| They can be executed using your Python3 interpreter or pytest.
