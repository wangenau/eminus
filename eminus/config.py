# SPDX-FileCopyrightText: 2021 The eminus developers
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
        self.use_torch = True  # Use the faster Torch FFTs if available
        self.use_gpu = False  # Disable GPU by default, since it is slower in my tests
        self.use_pylibxc = True  # Use Libxc over PySCF if available since it is faster
        self.threads = None  # Read threads from environment variables by default
        self.verbose = 'INFO'  # Only display warnings (and worse) by default

    # ### Class properties ###

    @property
    def use_torch(self):
        """Whether to use Torch or SciPy if Torch is installed."""
        # Add the logic in the getter method so it does not run on initialization since importing
        # Torch is rather slow
        if self._use_torch:
            try:
                import torch  # noqa: F401

                return True
            except ImportError:
                pass
        return False

    @use_torch.setter
    def use_torch(self, value):
        self._use_torch = value

    @property
    def use_gpu(self):
        """Whether to use Torch on the GPU if available."""
        # Only use GPU if Torch is available
        if self.use_torch and self._use_gpu:
            import torch

            return torch.cuda.is_available()
        return False

    @use_gpu.setter
    def use_gpu(self, value):
        self._use_gpu = value

    @property
    def use_pylibxc(self):
        """Whether to use pylibxc or PySCF for functionals if both are installed."""
        if self._use_pylibxc:
            try:
                import pylibxc  # noqa: F401

                return True
            except ImportError:
                pass
        return False

    @use_pylibxc.setter
    def use_pylibxc(self, value):
        self._use_pylibxc = value

    @property
    def threads(self):
        """Number of threads used in FFT calculations."""
        if self._threads is None:
            try:
                if self.use_torch:
                    import torch

                    return torch.get_num_threads()
                # Read the OMP threads for the default operators
                return int(os.environ['OMP_NUM_THREADS'])
            except KeyError:
                return None
        return int(self._threads)

    @threads.setter
    def threads(self, value):
        self._threads = value
        if isinstance(value, numbers.Integral):
            if self.use_torch:
                import torch

                return torch.set_num_threads(value)
            os.environ['OMP_NUM_THREADS'] = str(value)
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
        print('--- Configuration infos ---')
        print(f'Global verbosity : {self.verbose}')
        # Only print if PySCF or pylibxc is installed
        if not self.use_pylibxc:
            try:
                import pyscf  # noqa: F401

                print('Libxc backend    : PySCF')
            except ImportError:
                pass
        else:
            print('Libxc backend    : pylibxc')

        print('\n--- Performance infos ---')
        print(f'FFT backend : {"Torch" if self.use_torch else "SciPy"}')
        print(f'FFT device  : {"GPU" if self.use_gpu else "CPU"}')
        # Do not print threading information when using GPU
        if self.use_gpu:
            return
        # Check FFT threads
        if self.threads is None:
            print(
                'FFT threads : 1\n'
                'INFO: No OMP_NUM_THREADS environment variable was found.\nTo improve '
                'performance, add "export OMP_NUM_THREADS=n" to your ".bashrc".\nMake sure to '
                'replace "n", typically with the number of cores your CPU.\nTemporarily, you can '
                'set them in your Python environment with "eminus.config.threads=n".'
            )
        else:
            print(f'FFT threads : {self.threads}')


# Do not initialize the class when Sphinx is running
# Since we set the class instance to the module name Sphinx will only document the main docstring of
# the class without the properties
if 'sphinx-build' not in pathlib.Path(sys.argv[0]).name:
    sys.modules[__name__] = ConfigClass()
