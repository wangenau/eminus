.. _further:

Further information
*******************

- The source code for eminus is openly available at `GitLab <https://gitlab.com/nextdft/eminus>`_.
- The code has been described in a `master thesis <https://www.researchgate.net/publication/356537762_Domain-averaged_Fermi_holes_A_self-interaction_correction_perspective>`_.
- This thesis comes with an in-detail explanation of a minimalistic implementation called `SimpleDFT <https://gitlab.com/nextdft/simpledft>`_.
- For more information about implementing DFT using the `DFT++ <https://arxiv.org/abs/cond-mat/9909130>`_ formulation, fantastic lectures from Tomas Arias can be found `here <https://jdftx.org/PracticalDFT.html>`_.
- The code `JDFTx <https://jdftx.org/index.html>`_ from Ravishankar Sundararaman et al. offers a great reference implementation in C++ using DFT++.
- Another great read is the book by Fadjar Fathurrahman et al. called `Implementing Density Functional Theory Calculations <https://github.com/f-fathurrahman/ImplementingDFT>`_.
- This book outlines an implementation in Julia using a different formulation called `PWDFT.jl <https://github.com/f-fathurrahman/PWDFT.jl>`_.
- More open-source codes related to DFT, SIC, and more can be found on the `NextDFT <https://nextdft.gitlab.io/nextdft/>`_ and `OpenSIC <https://opensic.gitlab.io/opensic/>`_ pages.

Development
===========

To apply changes to the code without reinstalling the code, install eminus with

.. code-block:: bash

   git clone https://gitlab.com/nextdft/eminus.git
   cd eminus
   pip install -e .

Testing
-------

| To verify that changes work as intended, tests can be found in the `tests folder <https://gitlab.com/nextdft/eminus/-/tree/master/tests>`_.
| They can be executed using the Python interpreter or with `pytest <https://docs.pytest.org/>`_.
| pytest can be installed and executed with

.. code-block:: bash

   pip install pytest
   pytest

Linting
-------

| This code is lint-checked with `flake8 <https://flake8.pycqa.org/>`_, using a custom `configuration file <https://gitlab.com/nextdft/eminus/-/tree/master/.flake8>`_.
| To install flake8 and do a lint check, use

.. code-block:: bash

   pip install flake8 flake8-docstrings
   flake8

Documentation
-------------
| The documentation is automatically generated with `Sphinx <https://www.sphinx-doc.org/>`_, using a custom theme called `Furo <https://pradyunsg.me/furo/>`_.
| Both packages can be installed and the webpage can be built with

.. code-block:: bash

   pip install sphinx furo
   sphinx-build -b html ./docs ./public
