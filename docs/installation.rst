..
   SPDX-FileCopyrightText: 2021 The eminus developers
   SPDX-License-Identifier: Apache-2.0

.. _installation:

Installation
************

| The code is written for Python 3.8+.
| The following packages are needed for a minimum working installation

* `NumPy <https://numpy.org>`_
* `SciPy <https://scipy.org>`_

| All packages used have `OSI-approved <https://opensource.org/licenses/alphabetical>`_ licenses and are publicly visible.
| The code is tested to run under Ubuntu, Debian, macOS, Windows, and NixOS.

Installation with pip
=====================

The `package <https://pypi.org/project/eminus>`_ and all necessary dependencies can be installed with

.. code-block:: console

   pip install eminus

Depending on your operating system you may have to create a virtual environment.
This can be done, e.g., with

.. code-block:: console

   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install eminus

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

Weekly builds
=============

Weekly builds of the eminus :code:`dev` branch are created every Monday morning at 08:00 (UTC) and are uploaded to the `GitLab package registry <https://gitlab.com/wangenau/eminus/-/packages>`_.
The builds use the following version scheme: :code:`version.devYYYMMDD`, e.g., :code:`3.0.0.dev20241114`.
To install the weekly build, use the following command

.. code-block:: console

   pip install eminus -U --index-url https://gitlab.com/api/v4/projects/31610189/packages/pypi/simple

:code:`-U` will upgrade you to the latest eminus pre-release version, while :code:`--index-url` connects you to the eminus package registry.

If the current stable version is already installed, please uninstall it before switching to weekly builds, since the stable version will be preferred when upgrading.
You can uninstall eminus with

.. code-block:: console

   pip uninstall eminus

Docker image
============

To use a containerized version of the code, a `Docker container <https://hub.docker.com/r/wangenau/eminus>`_ has been created with all extras installed.
The following command starts the container and a Jupyter notebook server

.. code-block:: console

    docker run -it -p 8888:8888 wangenau/eminus:version

Opening the displayed URL in a browser will open the Jupyter environment.
Make sure to replace :code:`version` with the version you want to use.

You can also pass command line arguments to the container, e.g., to start a Python environment

.. code-block:: console

    docker run -it wangenau/eminus:version python

Nix usage
=========

To use the package under `Nix <https://nixos.org/>`_ one can easily create a development shell with all dependencies and (almost) all extras available. To do so, run the following commands on your Nix machine

.. code-block:: console

   git clone https://gitlab.com/wangenau/eminus.git
   cd eminus
   nix develop
