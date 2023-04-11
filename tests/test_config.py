#!/usr/bin/env python3
'''Test configuration class.'''
import inspect
import os
import pathlib

import pytest

import eminus
from eminus import config


def test_singleton():
    '''Check that config is truly a singleton.'''
    assert id(config) == id(eminus.config)


@pytest.mark.parametrize('level', ['debug', 0, 9])
def test_logger(level):
    '''Check that the logger gets properly updated.'''
    config.verbose = level
    assert config.verbose == eminus.log.verbose


def test_torch():
    '''Check the Torch initialization.'''
    try:
        import torch  # noqa: F401
        assert config.use_torch
    except ImportError:
        assert not config.use_torch


def test_libxc():
    '''Check the Libxc initialization.'''
    try:
        import pylibxc  # noqa: F401
        assert config.use_pylibxc
    except ImportError:
        assert not config.use_pylibxc


def test_threads():
    '''Check the threads setting.'''
    assert isinstance(config.threads, int)

    threads = 2
    if config.use_torch:
        os.environ['MKL_NUM_THREADS'] = str(threads)
    else:
        os.environ['OMP_NUM_THREADS'] = str(threads)
    assert config.threads == threads

    threads = 6
    config.threads = threads
    assert config.threads == threads


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
