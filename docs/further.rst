..
   SPDX-FileCopyrightText: 2021 The eminus developers
   SPDX-License-Identifier: Apache-2.0

.. _further:

Further information
*******************

- The code has been described in a `master thesis <https://researchgate.net/publication/356537762_Domain-averaged_Fermi_holes_A_self-interaction_correction_perspective>`_.
- This thesis comes with an in-detail explanation of a minimalistic implementation called `SimpleDFT <https://gitlab.com/wangenau/simpledft>`_.
- There is also a version of SimpleDFT written in Julia, called `SimpleDFT.jl <https://gitlab.com/wangenau/simpledft.jl>`_.
- A separate repository called `eminus-benchmarks <https://gitlab.com/wangenau/eminus-benchmarks>`_ is available for performance and code comparisons.
- A comparison of both SimpleDFT codes is available under the `SimpleDFT pages <https://wangenau.gitlab.io/simpledft_pages/>`_.
- For more information about implementing DFT using the `DFT++ <https://arxiv.org/abs/cond-mat/9909130>`_ formulation, fantastic lectures from Tomas Arias can be found `here <https://jdftx.org/PracticalDFT.html>`_.
- The code `JDFTx <https://jdftx.org/index.html>`_ from Ravishankar Sundararaman et al. offers a great reference implementation in C++ using DFT++.
- Another great read is the book by Fadjar Fathurrahman et al. called `Implementing Density Functional Theory Calculations <https://github.com/f-fathurrahman/ImplementingDFT>`_.
- This book outlines an implementation in Julia using a different formulation called `PWDFT.jl <https://github.com/f-fathurrahman/PWDFT.jl>`_.
- More open-source codes related to DFT, SIC, and more can be found on the `ESP <https://esp42.gitlab.io>`_ and `OpenSIC <https://opensic.gitlab.io/opensic>`_ pages.

Contact
=======

- If you encounter issues, please use the `GitLab issue tracker <https://gitlab.com/wangenau/eminus/-/issues>`_ file in bug reports.
- Additionally, there is a `Discord server <https://discord.gg/k2XwdMtVec>`_ where you can easily ask your questions.
- If there are still open questions about anything, feel free to `contact me <mailto:wangenau@protonmail.com>`_.

Development
===========

To apply changes to the code without reinstalling the code, install eminus with

.. code-block:: console

   git clone https://gitlab.com/wangenau/eminus.git
   cd eminus
   pip install -e .

To install all packages needed for development as listed below, use the following option

.. code-block:: console

   pip install -e .[dev]

Testing
-------

| To verify that changes work as intended, tests can be found in the `tests folder <https://gitlab.com/wangenau/eminus/-/tree/main/tests>`_.
| They can be executed using the Python interpreter or with `pytest <https://docs.pytest.org>`_.
| pytest can be installed and executed with

.. code-block:: console

   pip install pytest
   pytest

Coverage
--------

| To check that all functions have a test case a coverage report can be created.
| To install the necessary pytest extensions, use

.. code-block:: console

   pip install coverage pytest

| A detailed coverage report that gives a clear overview of the covered code `is uploaded with every release <https://wangenau.gitlab.io/eminus/htmlcov>`_.
| To create this webpage, use

.. code-block:: console

   coverage run -m pytest
   coverage html

| To only create the total coverage percentage value for all files, use

.. code-block:: console

   coverage run -m pytest
   coverage report

Linting and formatting
----------------------

| This code is lint-checked and formatted with `Ruff <https://beta.ruff.rs>`_, using a custom `style configuration <https://gitlab.com/wangenau/eminus/-/tree/main/pyproject.toml>`_.
| To install Ruff and do a lint check, use

.. code-block:: console

   pip install ruff
   ruff check .

To format the code use

.. code-block:: console

   ruff format .

Type checking
-------------

| This code is type-checked with `mypy <https://mypy-lang.org/>`_.
| To install mypy and do a static type check, use

.. code-block:: console

   pip install mypy
   mypy .

Documentation
-------------
| The documentation is automatically generated with `Sphinx <https://www.sphinx-doc.org>`_, using a custom theme called `Furo <https://pradyunsg.me/furo>`_.
| Both packages can be installed and the webpage can be built with

.. code-block:: console

   pip install sphinx furo sphinx-design sphinxcontrib-bibtex
   sphinx-build -b html ./docs ./public

The documentation build time can be shortened by using more processes, e.g., with

.. code-block:: console

   sphinx-build -j $(nproc) -b html ./docs ./public
