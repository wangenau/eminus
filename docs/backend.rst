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

Writing backend agnostic code is most of the time straightforward.
Where one normally uses NumPy, e.g., with

.. code-block:: python

   import numpy as np
   np.ones(3)

one can now write

.. code-block:: python

   from eminus import backend as xp
   xp.ones(3)

The backend can be changed using the :code:`config` module with

.. code-block:: python

   import eminus
   eminus.backend = "numpy"

Currently supported backends are NumPy and Torch.

There are a few differences between the two backends.
Luckily, many can be mitigated using the `array-api-compat <https://data-apis.org/array-api-compat>`_ package.
In cases where this does not work, a wrapper function can be added to the backend module, e.g., to get :code:`xp.delete` to work for Torch tensors.

The largest difference can be found in the strictness of types.
Where NumPy easily casts types, e.g., in a matrix multiplication between a float and a complex type, Torch needs an additional cast from float to complex.

To simplify some operations, there are also helper functions available in the :code:`backend` module, e.g., to check if a variable is a backend array

.. code-block:: python

   from eminus import backend as xp
   a = xp.ones(3)
   xp.is_array(a)

GPU
---

Torhc supports calculations on the GPU, while NumPy does not.
However, some functionalities are exclusive to NumPy (mostly in the :code:`extras`).
Therefore, it is sometimes needed to convert the Torch GPU tensors to NumPy CPU arrays.
For this, the :code:`to_np` helper function can be utilized

.. code-block:: python

   from eminus import backend as xp
   from eminus import config

   config.use_gpu = True
   a = xp.ones(3)
   a = xp.to_gpu(a)

Running tests
-------------

To test the functionality for all backends, each test will be performed for every supported tensors backend installed.
This simplifies the writing of tests but may result in redundant tests.
The tests will be performed consecutively.
If there are failures in one backend, the other backend will not be performed.
