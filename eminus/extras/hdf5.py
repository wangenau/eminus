# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""HDF5 file handling.

All necessary dependencies to use this extra can be installed with::

    pip install eminus[hdf5]
"""

import numpy as np

from ..io.json import _custom_object_hook
from ..logger import log


def read_hdf5(filename):
    """Load objects from an HDF5 file.

    Args:
        filename: HDF5 input file path/name.

    Returns:
        Class object.
    """
    try:
        from h5py import _hl, check_string_dtype, File
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[hdf5]".\n\n'
        )
        raise

    if not filename.endswith((".h5", ".hdf", ".hdf5")):
        filename += ".hdf5"

    def read_hdf5_recursively(fh, path):
        """Load HDF5 files while creating the appropriate nested object structure."""
        dct = {}
        for key, value in fh[path].items():
            if isinstance(value, _hl.dataset.Dataset):
                # Restore None values using attributes
                if "None" in value.attrs:
                    dct[key] = None
                # Strings are serialized as bytes objects, restore them
                elif check_string_dtype(value.dtype):
                    dct[key] = value.asstr()[()]
                    # Lists of strings are encoded as arrays, restore them as well
                    if isinstance(dct[key], np.ndarray):
                        dct[key] = dct[key].tolist()
                else:
                    dct[key] = value[()]
            elif isinstance(value, _hl.group.Group):
                dct[key] = read_hdf5_recursively(fh, f"{path}{key}/")
        # Create eminus objects from dictionaries, reuse the JSON helper function
        return _custom_object_hook(dct)

    with File(filename) as fh:
        return read_hdf5_recursively(fh, "/")


def write_hdf5(obj, filename, compression="gzip", compression_opts=4):
    """Save objects in an HDF5 file.

    Args:
        obj: Class object.
        filename: HDF5 output file path/name.

    Keyword Args:
        compression: Compression filter.
        compression_opts: Compression level.
    """
    try:
        from h5py import File
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[hdf5]".\n\n'
        )
        raise

    import eminus

    if not filename.endswith((".h5", ".hdf", ".hdf5")):
        filename += ".hdf5"

    def write_hdf5_recursively(fp, path, dic):
        """Save HDF5 files with nested object structures in mind."""
        for key, value in dic.items():
            # Make a dictionary out of eminus objects
            if isinstance(
                value,
                (
                    eminus.Atoms,
                    eminus.SCF,
                    eminus.energies.Energy,
                    eminus.gth.GTH,
                    eminus.kpoints.KPoints,
                    eminus.occupations.Occupations,
                ),
            ):
                write_hdf5_recursively(fp, f"{path}{key}/", value.__dict__)
            # Dictionaries are not storable, create a group for every dictionary (and eminus object)
            elif isinstance(value, dict):
                write_hdf5_recursively(fp, f"{path}{key}/", value)
            # None values can not be stored in HDF5 files, set an attribute for them
            # The logger class is not serializable, just set it to None as well
            elif isinstance(value, eminus.logger.CustomLogger) or value is None:
                dataset = fp.create_dataset(f"{path}{key}", data=[])
                dataset.attrs["None"] = True
            # Compress arrays
            elif isinstance(value, np.ndarray):
                fp.create_dataset(
                    f"{path}{key}",
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                fp.create_dataset(f"{path}{key}", data=value)

    with File(filename, "w") as fp:
        write_hdf5_recursively(fp, "/", obj.__dict__)
