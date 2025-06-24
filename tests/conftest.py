# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Initialization hooks to modify pytest runs."""

import os
import subprocess
import sys

import eminus


def pytest_sessionstart(session):
    """Hook that runs at the start of the pytest session.

    Runs pytest depending on an environment variable.
    """
    if os.environ.get("PYTEST_RERUN_TORCH") != "1":
        eminus.config.backend = "numpy"
    else:
        eminus.config.backend = "torch"

    terminalreporter = session.config.pluginmanager.getplugin("terminalreporter")
    terminalreporter.section(f"{eminus.config.backend} backend tests")


def pytest_sessionfinish(session, exitstatus):
    """Hook that runs at the end of the pytest session.

    Rerun pytest using a different backend.
    """
    # Only run followup tests when all tests passed, otherwise we will see no traceback
    if os.environ.get("PYTEST_RERUN_TORCH") != "1" and exitstatus == 0:
        try:
            import array_api_compat  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pass
        else:
            terminalreporter = session.config.pluginmanager.getplugin("terminalreporter")
            terminalreporter.write_line("")  # Fix printing of consecutive backend sections
            os.environ["PYTEST_RERUN_TORCH"] = "1"
            # We need to create a new process
            # Otherwise the precalculated values in some tests will not get updated
            retcode = subprocess.call([sys.executable, "-m", "pytest", *sys.argv[1:]])  # noqa: S603
            sys.exit(retcode)
