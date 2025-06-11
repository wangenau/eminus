# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Consolidated configuration module."""

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
        self.backend = "torch"  # Use faster Torch FFTs from a different backend if available
        self.use_pylibxc = True  # Use Libxc over PySCF if available since it is faster
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
                import torch
            except ImportError:
                self._backend = "numpy"
            else:
                torch.set_default_dtype(torch.double)
        else:
            self._backend = "numpy"

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
        sys.stdout.write(f"Array backend    : {self.backend}\n")


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
    backend = ""  #: Whether to use SciPy or a different backend if installed.
    use_pylibxc = False  #: Whether to use pylibxc or PySCF for functionals if both are installed.
    verbose = ""  #: Logger verbosity level.

    def info():
        """Print configuration and performance information."""
        return
