#!/usr/bin/env python3
'''Logger initialization and configuration.'''
import copy
import logging
import sys


class CustomLogger(logging.Logger):
    '''Custom logger for the usage outside of classes.

    This is just a basic logger, but with an added verbose property.

    Args:
        name (str): Logger name.
    '''
    def __init__(self, name):
        super().__init__(name)

    @property
    def verbose(self):
        '''Verbosity level.'''
        return self._verbose

    @verbose.setter
    def verbose(self, level):
        self._verbose = get_level(level)
        self.setLevel(self._verbose)


class CustomFormatter(logging.Formatter):
    '''Custom logger formatter.'''
    def format(self, record):
        '''Use different formatting for different logging levels.

        Args:
            record: LogRecord object.
        '''
        if record.levelno >= logging.WARNING:
            # Print the level name for errors and warning
            self._style._fmt = '%(levelname)s: %(msg)s'
        else:
            # But not for infos and debug messages
            self._style._fmt = '%(msg)s'
        return super().format(record)


# The following code is not guarded by a function because it has to be run once the logger is called
# to set up the basic logger configuration

# Create a base logger that can be used outside of classes
logging.setLoggerClass(CustomLogger)
log = logging.getLogger('eminus')

# Basic logger setup
formatter = CustomFormatter()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logging.root.addHandler(handler)


def create_logger(object):
    '''Create a logger unique to an object.

    Args:
        object: Instance of a class.
    '''
    # Use ID of object to create a unique logger
    # Without this setting the verbosity in one instance would affect other instances
    local_log = logging.getLogger(str(id(object)))
    local_log.verbose = str(log.verbose)
    return local_log


def get_level(verbose):
    '''Validate logging levels.

    Args:
        verbose (int | str): Level of output (case insensitive).
    '''
    log_levels = {
        0: 'CRITICAL',
        1: 'ERROR',
        2: 'WARNING',
        3: 'INFO',
        4: 'DEBUG'
    }
    if verbose is None:
        return str(log.verbose)
    if isinstance(verbose, int):
        level = log_levels.get(verbose, 'DEBUG')
    else:
        level = verbose.upper()
    return level


def name(newname):
    '''Add a name to functions without evaluating them for better logging.

    Args:
        newname (str): Function name.

    Returns:
        Callable: Decorator.
    '''
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator
