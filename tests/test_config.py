# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test configuration class."""

import os

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


def test_backend():
    """Check the backend initialization."""
    try:
        import jax.numpy  # noqa: F401

        config.backend = "jax"
        assert config.backend == "jax"
    except ImportError:
        assert config.backend != "jax"

    try:
        import torch  # noqa: F401

        config.backend = "torch"
        assert config.backend == "torch"
    except ImportError:
        assert config.backend != "torch"


def test_libxc():
    """Check the Libxc initialization."""
    try:
        import pylibxc  # noqa: F401

        assert config.use_pylibxc
    except ImportError:
        assert not config.use_pylibxc


def test_threads():
    """Check the threads setting."""
    assert isinstance(config.threads, int) or config.threads is None

    threads = 2
    if config.backend == "torch":
        import torch

        torch.set_num_threads(threads)
    else:
        os.environ["OMP_NUM_THREADS"] = str(threads)
    assert config.threads == threads
    assert isinstance(config.threads, int)

    threads = 6
    config.threads = threads
    assert config.threads == threads
    assert isinstance(config.threads, int)


def test_info():
    """Check that the config info function properly executes."""
    config.info()


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
