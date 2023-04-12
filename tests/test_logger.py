#!/usr/bin/env python3
'''Test logging class.'''
import pytest

import eminus
from eminus import logger


def test_singleton():
    '''Check that log is truly a singleton.'''
    assert id(logger.log) == id(eminus.log)


def test_independence():
    '''Check that loggers do not influence each other.'''
    class Tmp:
        pass
    tmp = Tmp()
    log = logger.create_logger(tmp)
    assert log.verbose == logger.log.verbose
    log.verbose = 'critical'
    assert log.verbose != logger.log.verbose


@pytest.mark.parametrize('level, ref', [('DEBUG', 'DEBUG'),
                                        ('debug', 'DEBUG'),
                                        (4, 'DEBUG'),
                                        (0, 'CRITICAL'),
                                        (None, logger.log.verbose),
                                        (9, 'DEBUG')])
def test_level(level, ref):
    '''Test logging levels.'''
    class Tmp:
        pass
    tmp = Tmp()
    log = logger.create_logger(tmp)
    log.verbose = level
    assert log.verbose == ref


def test_name():
    '''Test the name changing decorator.'''
    name = 'newname'

    @logger.name(name)
    def tmp():
        pass
    assert tmp.__name__ == name


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
