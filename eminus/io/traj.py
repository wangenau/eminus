# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""TRAJ file handling."""

import numpy as np

from ..logger import log
from ..units import ang2bohr
from .xyz import write_xyz


def read_traj(filename):
    """Load atom species and positions from TRAJ files.

    TRAJ files are just multiple XYZ files appended to one file.
    See :func:`~eminus.io.xyz.read_xyz` for more information about the XYZ file format.

    Args:
        filename: TRAJ input file path/name.

    Returns:
        Atom species and positions.
    """
    if not filename.endswith((".trj", ".traj")):
        filename += ".traj"

    with open(filename, encoding="utf-8") as fh:
        lines = fh.readlines()
        Nlines = len(lines)

        # The first line contains the number of atoms
        Natoms = int(lines[0].strip())

        # The second line can contain a comment, print it if available
        comment = lines[1].strip()
        if comment:
            log.info(f'TRAJ file comment: "{comment}"')

        traj = []
        for frame in range(Nlines // (2 + Natoms)):
            atom = []
            pos = []
            # Following lines contain atom positions with the format: Atom x-pos y-pos z-pos
            for line in lines[(2 + Natoms) * frame + 2 : (2 + Natoms) * (frame + 1)]:
                line_split = line.strip().split()
                atom.append(line_split[0])
                pos.append(np.float64(line_split[1:4]))
            # XYZ files are in Angstrom, so convert to Bohr
            pos = ang2bohr(np.asarray(pos))
            traj.append((atom, pos))
    return traj


def write_traj(obj, filename, fods=None, elec_symbols=("X", "He")):
    """Generate TRAJ files from atoms objects.

    TRAJ files are just multiple XYZ files appended to one file.
    See :func:`~eminus.io.xyz.write_xyz` for more information about the XYZ file format.

    Args:
        obj: Atoms or SCF object or list/tuple of these objects.
        filename: TRAJ output file path/name.

    Keyword Args:
        fods: FOD coordinates to write.
        elec_symbols: Identifier for up and down FODs.
    """
    if not filename.endswith((".trj", ".traj")):
        filename += ".traj"

    if isinstance(obj, (list, tuple)):
        for iobj in obj:
            write_xyz(iobj, filename, fods=fods, elec_symbols=elec_symbols, trajectory=True)
    else:
        write_xyz(obj, filename, fods=fods, elec_symbols=elec_symbols, trajectory=True)
