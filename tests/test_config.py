# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test configuration class."""

import pytest

import eminus
from eminus import config


def test_singleton():
    """Check that config is truly a singleton."""
    assert id(config) == id(eminus.config)


@pytest.mark.parametrize("level", ["debug", 0, 9])
def test_logger(level):
    """Check that the logger gets properly updated."""
    config.verbose = level
    assert config.verbose == eminus.log.verbose


def test_libxc():
    """Check the Libxc initialization."""
    try:
        import pylibxc  # noqa: F401

        assert config.use_pylibxc
    except ImportError:
        assert not config.use_pylibxc


def test_info():
    """Check that the config info function properly executes."""
    config.info()


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
