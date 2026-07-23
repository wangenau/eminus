# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Hooks to modify pytest runs."""

import eminus


def pytest_addoption(parser):
    """Custom pytest arguments."""
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="Select a backend to run tests with.",
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        default=None,
        help="Use the GPU to perform tests.",
    )


def pytest_configure(config):
    """Hook that performs initial configurations."""
    backend = config.getoption("--backend")
    if backend is not None:
        eminus.config.backend = backend
    use_gpu = config.getoption("--gpu")
    if use_gpu is not None:
        eminus.config.use_gpu = use_gpu


def pytest_sessionstart(session):
    """Hook that runs at the start of the pytest session."""
    terminalreporter = session.config.pluginmanager.getplugin("terminalreporter")
    device = "GPU" if eminus.config.use_gpu else "CPU"
    terminalreporter.write_sep("-", f"Run tests on {device} with {eminus.config.backend}")
