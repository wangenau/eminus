..
   SPDX-FileCopyrightText: 2025 The eminus developers
   SPDX-License-Identifier: Apache-2.0

.. _backend:

Backend
*******

The eminus code is written such that multiple array backends can be used.
Naturally, this adds a layer of complexity.
Luckily, not too much.
This document will inform developers about possible intricacies.

Writing backend agnostic code
=============================

Writing backend agnostic code is most of the time quite straightforward.
Where one normally uses NumPy, e.g., with

.. code-block:: python

   import numpy as np
   np.ones(3)

one can now write

.. code-block:: python

   from eminus import backend as xp
   xp.ones(3)

One caveat is that one has to type imports of submodules explicitly instead of relying on :code:`from` imports.

.. code-block:: python

   from eminus import backend as xp
   xp.linalg.det([[3]])

This is because the backend module will be determined at runtime.
:code:`from` imports do not support this dynamic system.

The backend can be changed using the :code:`config` module with

.. code-block:: python

   import eminus
   eminus.backend = "numpy"

Currently supported backends are NumPy, CuPy, and Torch.

There are a few differences between those backends.
Luckily, many can be mitigated using the `array-api-compat <https://data-apis.org/array-api-compat>`_ package.
In cases where this does not work, a wrapper function can be added to the backend module, e.g., to get :code:`xp.delete` to work for Torch tensors.

The largest difference can be found in the strictness of types.
Where NumPy easily casts types, e.g., in a matrix multiplication between a float and a complex type, CuPy and Torch often need an additional cast from float to complex.

To simplify some operations, there are also helper functions available in the :code:`backend` module, e.g., to check if a variable is a backend array

.. code-block:: python

   from eminus import backend as xp
   a = xp.ones(3)
   xp.is_array(a)

GPU
---

CuPy and Torch support calculations on the GPU, while NumPy does not.
However, some functionalities are exclusive to NumPy (mostly in the :code:`extras`).
Therefore, it is sometimes needed to convert the GPU arrays to NumPy CPU arrays.
For this, the :code:`to_np` helper function can be utilized

.. code-block:: python

   from eminus import backend as xp
   from eminus import config

   config.use_gpu = True
   a = xp.ones(3)
   a = xp.to_np(a)

Running tests
-------------

To test the functionality for a backend, one can select it by using the custom pytest :code:`--backend` option.
By default, the tests will be performed using NumPy.
Accordingly, to test different devices, there exists the pytest :code:`--gpu` option, which is not added by default.
The tests are written such that they work independently of the selected backend.
To test arrays that may exist on different devices or as arrays of different backends, one can use the utility functions in the :code:`testing` module.
For running tests with different backends using Nix, one can use different developer shells, e.g., with :code:`--nix develop .#torch`.
