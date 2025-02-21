# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""File input and output functionalities."""

from .cube import read_cube, write_cube
from .gth import read_gth
from .json import read_json, write_json
from .pdb import create_pdb_str, write_pdb
from .poscar import read_poscar, write_poscar
from .traj import read_traj, write_traj
from .xyz import read_xyz, write_xyz

__all__ = [
    "create_pdb_str",
    "read",
    "read_cube",
    "read_gth",
    "read_json",
    "read_poscar",
    "read_traj",
    "read_xyz",
    "write",
    "write_cube",
    "write_json",
    "write_pdb",
    "write_poscar",
    "write_traj",
    "write_xyz",
]


def read(filename, *args, **kwargs):
    """Unified file reader function.

    Args:
        filename: Input file path/name.
        *args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.

    Returns:
        Read file information.
    """
    if filename.endswith(".json"):
        return read_json(filename, *args, **kwargs)
    if filename.endswith((".h5", ".hdf", ".hdf5")):
        from ..extras import read_hdf5

        return read_hdf5(filename, *args, **kwargs)
    if filename.endswith(".xyz"):
        return read_xyz(filename, *args, **kwargs)
    if "POSCAR" in filename:
        return read_poscar(filename, *args, **kwargs)
    if filename.endswith((".trj", ".traj")):
        return read_traj(filename, *args, **kwargs)
    if filename.endswith((".cub", ".cube")):
        return read_cube(filename, *args, **kwargs)
    msg = "File ending not recognized."
    raise NotImplementedError(msg)


def write(obj, filename, *args, **kwargs):  # noqa: PLR0911
    """Unified file writer function.

    Args:
        obj: Class object.
        filename: Input file path/name.
        *args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.

    Returns:
        None.
    """
    if filename.endswith(".json"):
        return write_json(obj, filename, *args, **kwargs)
    if filename.endswith((".h5", ".hdf", ".hdf5")):
        from ..extras import write_hdf5

        return write_hdf5(obj, filename, *args, **kwargs)
    if filename.endswith(".xyz"):
        return write_xyz(obj, filename, *args, **kwargs)
    if "POSCAR" in filename:
        return write_poscar(obj, filename, *args, **kwargs)
    if filename.endswith((".trj", ".traj")):
        return write_traj(obj, filename, *args, **kwargs)
    if filename.endswith((".cub", ".cube")):
        return write_cube(obj, filename, *args, **kwargs)
    if filename.endswith(".pdb"):
        return write_pdb(obj, filename, *args, **kwargs)
    msg = "File ending not recognized."
    raise NotImplementedError(msg)
