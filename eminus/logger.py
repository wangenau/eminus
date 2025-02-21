# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Logger initialization and configuration."""

import logging
import numbers
import sys


class CustomLogger(logging.Logger):
    """Custom logger for the usage outside of classes.

    This is just a basic logger but with an added verbose property.

    Args:
        name: Logger name.
    """

    def __init__(self, name):
        """Initialize the CustomLogger object."""
        super().__init__(name)

    @property
    def verbose(self):
        """Verbosity level."""
        return self._verbose

    @verbose.setter
    def verbose(self, level):
        self._verbose = get_level(level)
        self.setLevel(self._verbose)


class CustomFormatter(logging.Formatter):
    """Custom logger formatter."""

    def format(self, record):
        """Use different formatting for different logging levels.

        Args:
            record: LogRecord object.

        Returns:
            Formatted log text.
        """
        if record.levelno >= logging.WARNING:
            # Print the level name for errors and warnings
            self._style._fmt = "%(levelname)s: %(msg)s"
        else:
            # But not for info and debug messages
            self._style._fmt = "%(msg)s"
        return super().format(record)


# The following code is not guarded by a function because it has to be run once the logger is called
# to set up the basic logger configuration

# Create a base logger that can be used outside of classes
logging.setLoggerClass(CustomLogger)
#: Global logging object.
log = logging.getLogger("eminus")

# Basic logger setup
__formatter = CustomFormatter()
__handler = logging.StreamHandler(sys.stdout)
__handler.setFormatter(__formatter)
logging.root.addHandler(__handler)


def create_logger(obj):
    """Create a logger unique to an object.

    Args:
        obj: Instance of a class.

    Returns:
        Logger object.
    """
    # Use the ID of objects to create a unique logger
    # Without this setting the verbosity in one instance would affect other instances
    local_log = logging.getLogger(str(id(obj)))
    local_log.verbose = log.verbose
    return local_log


def get_level(verbose):
    """Validate logging levels.

    Args:
        verbose: Level of output.

    Returns:
        Logging level.
    """
    log_levels = {
        0: "CRITICAL",
        1: "ERROR",
        2: "WARNING",
        3: "INFO",
        4: "DEBUG",
    }
    # Use the global logging level for None
    if verbose is None:
        level = log.verbose
    # Fall back to DEBUG if the level is not available
    elif isinstance(verbose, numbers.Number):
        level = log_levels.get(verbose, "DEBUG")
    else:
        level = verbose
    level = level.upper()
    if level not in log_levels.values():
        msg = f"{level} is no recognized logging level."
        raise ValueError(msg)
    return level


def name(newname):
    """Add a name to functions without evaluating them for better logging.

    Args:
        newname: Function name.

    Returns:
        Decorator.
    """

    def decorator(f):
        f.__name__ = newname
        return f

    return decorator
