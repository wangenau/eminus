# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Consolidated configuration module."""

import numbers
import os
import pathlib
import sys

from .logger import log


class ConfigClass:
    """Configuration class holding user specifiable variables.

    An instance of this class will be set as the same name as this module. This will effectively
    make this module a singleton data class.
    """

    def __init__(self):
        """Initialize the ConfigClass object."""
        self.backend = "torch"  # Use Torch as backend if available, default to NumPy otherwise
        self.use_gpu = False  # Disable GPU by default, since one may be restrictd by a small VRAM
        self.use_pylibxc = True  # Use Libxc over PySCF if available since it is faster
        self.threads = None  # Read threads from environment variables by default
        self.verbose = "INFO"  # Only display warnings (and worse) by default

    # ### Class properties ###

    @property
    def backend(self):
        """Whether to use NumPY or a different backend if installed."""
        return self._backend

    @backend.setter
    def backend(self, value):
        self._backend = value.lower()
        if self._backend == "torch":
            try:
                from array_api_compat import torch
            except ImportError:
                self._backend = "numpy"
            else:
                torch.set_default_dtype(torch.float64)
        else:
            self._backend = "numpy"

    @property
    def use_gpu(self):
        """Whether to use the GPU if available."""
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        if self.backend == "torch":
            import torch

            # When using set_default_device the whole runtime will use a device context manager
            if torch.cuda.is_available():
                if value:
                    torch.set_default_device("cuda")
                    self._use_gpu = True
                else:
                    torch.set_default_device("cpu")
                    self._use_gpu = False
            else:
                self._use_gpu = False
        else:
            self._use_gpu = False

    @property
    def use_pylibxc(self):
        """Whether to use pylibxc or PySCF for functionals if both are installed."""
        if self._use_pylibxc:
            try:
                import pylibxc  # noqa: F401
            except ImportError:
                pass
            else:
                return True
        return False

    @use_pylibxc.setter
    def use_pylibxc(self, value):
        self._use_pylibxc = value

    @property
    def threads(self):
        """Number of threads used in array calculations."""
        if self._threads is None:
            try:
                if self.backend == "torch":
                    from array_api_compat import torch

                    return torch.get_num_threads()
                # Read the OMP threads for the default operators
                return int(os.environ["OMP_NUM_THREADS"])
            except KeyError:
                return None
        return int(self._threads)

    @threads.setter
    def threads(self, value):
        self._threads = value
        if isinstance(value, numbers.Integral):
            if self.backend == "torch":
                from array_api_compat import torch

                return torch.set_num_threads(value)
            os.environ["OMP_NUM_THREADS"] = str(value)
        return None

    @property
    def verbose(self):
        """Logger verbosity level."""
        return log.verbose

    @verbose.setter
    def verbose(self, value):
        # Logic in setter to run it on initialization
        log.verbose = value

    # ### Class methods ###

    def info(self):
        """Print configuration and performance information."""
        sys.stdout.write("--- Configuration infos ---\n")
        sys.stdout.write(f"Global verbosity : {self.verbose}\n")
        # Only print if PySCF or pylibxc is installed
        if not self.use_pylibxc:
            try:
                import pyscf  # noqa: F401

                sys.stdout.write("Libxc backend    : PySCF\n")
            except ImportError:
                pass
        else:
            sys.stdout.write("Libxc backend    : pylibxc\n")

        sys.stdout.write(
            "\n--- Performance infos ---\n"
            f"Array backend    : {self.backend}\n"
            f"Array device     : {'GPU' if self.use_gpu else 'CPU'}\n"
        )
        # Do not print threading information when using GPU
        if self.use_gpu:
            return
        # Check threads
        if self.threads is None:
            sys.stdout.write(
                "Array threads    : 1\n"
                "INFO: No OMP_NUM_THREADS environment variable was found.\nTo improve "
                'performance, add "export OMP_NUM_THREADS=n" to your ".bashrc".\nMake sure to '
                'replace "n", typically with the number of cores your CPU.\nTemporarily, you can '
                'set them in your Python environment with "eminus.config.threads=n".\n'
            )
        else:
            sys.stdout.write(f"Array threads    : {self.threads}\n")


if (
    "sphinx-build" not in pathlib.Path(sys.argv[0]).name
    and "stubtest" not in pathlib.Path(sys.argv[0]).name
):
    # Do not initialize the class when Sphinx or stubtest is running
    # Since we set the class instance to the module name Sphinx would only document
    # the main docstring of the class without the properties
    sys.modules[__name__] = ConfigClass()
else:
    # Add mock variables for all properties and methods of the ConfigClass to the module
    # This allows IDEs to see that the module has said attribute
    # This also allows for stubtesting and documentation of these variables and functions
    backend = ""  #: Whether to use NumPy or a different backend if installed.
    use_gpu = False  #: Whether to use the GPU if available.
    use_pylibxc = False  #: Whether to use pylibxc or PySCF for functionals if both are installed.
    threads = 0  #: Number of threads used in array calculations.
    verbose = ""  #: Logger verbosity level.

    def info():
        """Print configuration and performance information."""
        return
