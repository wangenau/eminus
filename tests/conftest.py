# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Hooks to modify pytest runs."""

import eminus


def pytest_addoption(parser):
    """Backend selection argument."""
    parser.addoption(
        "--backend",
        action="store",
        default="numpy",
        help="Select a backend to run tests with.",
    )


def pytest_configure(config):
    """Hook that performs initial configurations."""
    backend = config.getoption("--backend")
    eminus.config.backend = backend


def pytest_sessionstart(session):
    """Hook that runs at the start of the pytest session."""
    terminalreporter = session.config.pluginmanager.getplugin("terminalreporter")
    terminalreporter.write_sep("-", f"{eminus.config.backend} backend tests")
