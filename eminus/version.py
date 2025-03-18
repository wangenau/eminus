# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Package version number and version info function."""

import importlib.metadata
import platform
import sys

__version__ = "3.0.3"
#: eminus ASCII logo.
LOGO = (" ___ _____ _ ___ _ _ ___ \n"
        "| -_|     | |   | | |_ -|\n"
        "|___|_|_|_|_|_|_|___|___|")  # fmt: skip


def info():
    """Print version numbers and availability of packages."""
    dependencies = ("numpy", "scipy")
    extras = ("dftd3", "h5py", "pyscf", "torch", "plotly", "nglview")
    dev = ("pylibxc", "mypy", "notebook", "pytest", "ruff", "sphinx")

    sys.stdout.write("\n".join([f"{line:>35}" for line in LOGO.split("\n")]))
    sys.stdout.write(
        "\n\neminus - Pythonic electronic structure theory"
        "\n https://doi.org/10.1016/j.softx.2025.102035\n"
        "\n--- Platform infos ---"
        f"\nPlatform   : {platform.system()} {platform.machine()}"
        f"\nRelease    : {platform.release()} {platform.version()}"
        "\n\n--- Version infos ---"
        f"\npython     : {sys.version.split()[0]}"
        f"\neminus     : {__version__}\n"
    )
    for pkg in dependencies + extras + dev:
        try:
            sys.stdout.write(f"{pkg:<11}: {importlib.metadata.version(pkg)}\n")
        except ModuleNotFoundError:  # noqa: PERF203
            if pkg in dependencies:
                sys.stdout.write(f"{pkg:<11}: Dependency not installed\n")
            elif pkg in extras:
                sys.stdout.write(f"{pkg:<11}: Extra not installed\n")
