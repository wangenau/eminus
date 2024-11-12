# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""HDF5 file handling.

All necessary dependencies to use this extra can be installed with::

    pip install eminus[hdf5]
"""

import json

from ..io.json import _custom_object_hook, _CustomEncoder
from ..logger import log


def read_hdf5(filename):
    """Load objects from a HDF5 file.

    Args:
        filename: HDF5 input file path/name.

    Returns:
        Class object.
    """
    try:
        from h5py import File
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[hdf5]".\n\n'
        )
        raise

    if not filename.endswith((".h5", ".hdf", ".hdf5")):
        filename += ".hdf5"

    with File(filename) as fh:
        return json.loads(fh["eminus_object"][()], object_hook=_custom_object_hook)


def write_hdf5(obj, filename):
    """Save objects in a HDF5 file.

    Args:
        obj: Class object.
        filename: HDF5 output file path/name.
    """
    try:
        from h5py import File
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[hdf5]".\n\n'
        )
        raise

    if not filename.endswith((".h5", ".hdf", ".hdf5")):
        filename += ".hdf5"

    with File(filename, "w") as fp:
        fp.create_dataset("eminus_object", data=json.dumps(obj, cls=_CustomEncoder))
