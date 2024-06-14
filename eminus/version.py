# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Package version number and version info function."""

import importlib
import platform
import sys

__version__ = '2.7.1'
#: eminus ASCII logo.
LOGO = (' ___ _____ _ ___ _ _ ___ \n'
        '| -_|     | |   | | |_ -|\n'
        '|___|_|_|_|_|_|_|___|___|\n')  # fmt: skip


def info():
    """Print version numbers and availability of packages."""
    dependencies = ('numpy', 'scipy')
    extras = ('torch', 'pyscf', 'dftd3', 'plotly', 'nglview')
    dev = ('matplotlib', 'notebook', 'pylibxc', 'pytest', 'sphinx', 'furo')

    print(LOGO)
    print(
        '--- Platform infos ---'
        f'\nPlatform   : {platform.system()} {platform.machine()}'
        f'\nRelease    : {platform.release()} {platform.version()}'
        '\n\n--- Version infos ---'
        f'\npython     : {sys.version.split()[0]}'
        f'\neminus     : {__version__}'
    )
    for pkg in dependencies + extras + dev:
        try:
            module = importlib.import_module(pkg)
            try:
                print(f'{pkg:<11}: {module.__version__}')
            except AttributeError:
                # pylibxc does not use the standard version identifier
                print(f'{pkg:<11}: {module.version.__version__}')
        except ModuleNotFoundError:  # noqa: PERF203
            if pkg in dependencies:
                print(f'{pkg:<11}: Dependency not installed')
            elif pkg in extras:
                print(f'{pkg:<11}: Extra not installed')
