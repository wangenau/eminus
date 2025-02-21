# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""CUBE file handling."""

import textwrap
import time

import numpy as np

from ..data import NUMBER2SYMBOL, SYMBOL2NUMBER
from ..logger import log
from ..version import __version__


def read_cube(filename):
    """Load atom and cell data from CUBE files.

    There is no standard for CUBE files. The following format has been used.
    File format definition: https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html

    Args:
        filename: CUBE input file path/name.

    Returns:
        Species, positions, charges, cell size, sampling, and field array.
    """
    if not filename.endswith((".cub", ".cube")):
        filename += ".cube"

    # Atomic units and a cell that starts at (0,0,0) are assumed.
    with open(filename, encoding="utf-8") as fh:
        lines = fh.readlines()

        # The first and second lines can contain comments, print them if available
        comment = f"{lines[0].strip()}\n{lines[1].strip()}"
        if comment:
            log.info(f'CUBE file comment: "{comment}"')

        # Lines 4 to 6 contain the sampling per axis and the cell basis vectors with length a/s
        s = np.empty(3, dtype=int)
        a = np.empty((3, 3))
        for i, line in enumerate(lines[3:6]):
            line_split = line.strip().split()
            s[i] = line_split[0]
            a[i] = s[i] * np.float64(line_split[1:])

        atom = []
        pos = []
        Z = []
        # Following lines contain atom positions with the format: atom-id charge x-pos y-pos z-pos
        _offset = 0
        for _offset, line in enumerate(lines[6:]):
            line_split = line.strip().split()
            # If the first value is not a (positive) integer, we have reached the field data
            if not line_split[0].isdigit():
                break
            atom.append(NUMBER2SYMBOL[int(line_split[0])])
            Z.append(float(line_split[1]))
            pos.append(np.float64(line_split[2:5]))
    pos = np.asarray(pos)

    # The rest of the data is the field data
    # Split the strings, flatten the lists of lists, and convert to a float numpy array
    tmp_list = [l.split() for l in lines[6 + _offset :]]
    field_list = [item for sublist in tmp_list for item in sublist]
    field = np.asarray(field_list, dtype=float)
    return atom, pos, Z, a, s, field


def write_cube(obj, filename, field, fods=None, elec_symbols=("X", "He")):
    """Generate CUBE files from given field data.

    There is no standard for CUBE files. The following format has been used to work with VESTA.
    File format definition: https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html

    Args:
        obj: Atoms or SCF object.
        filename: CUBE output file path/name.
        field: Real-space field data.

    Keyword Args:
        fods: FOD coordinates to write.
        elec_symbols: Identifier for up and down FODs.
    """
    # Atomic units are assumed, so there is no need for conversion.
    atoms = obj._atoms

    if not filename.endswith((".cub", ".cube")):
        filename += ".cube"

    if "He" in atoms.atom and atoms.unrestricted:
        log.warning(
            'You need to modify "elec_symbols" to write helium with FODs in the spin-'
            "polarized case."
        )

    # Make sure we have real-valued data in the correct order
    if field is None:
        log.warning('The provided field is "None".')
        return
    field = np.real(field)

    with open(filename, "w", encoding="utf-8") as fp:
        # The first two lines have to be a comment
        # Print information about the file and program, and the file creation time
        fp.write(f"File generated with eminus {__version__} on {time.ctime()}\n\n")
        # Number of atoms (int), and origin of the coordinate system (float)
        # The origin is normally at 0,0,0 but we could move our box, so take the minimum
        if fods is None:
            fp.write(f"{atoms.Natoms}  ")
        else:
            fp.write(f"{atoms.Natoms + sum(len(i) for i in fods)}  ")
        fp.write("0.0  0.0  0.0\n")
        # Number of points per axis (int), and vector defining the axis (float)
        fp.write(
            f"{atoms.s[0]}  {atoms.a[0, 0] / atoms.s[0]:.6f}  {atoms.a[0, 1] / atoms.s[0]:.6f}"
            f"  {atoms.a[0, 2] / atoms.s[0]:.6f}\n"
            f"{atoms.s[1]}  {atoms.a[1, 0] / atoms.s[1]:.6f}  {atoms.a[1, 1] / atoms.s[1]:.6f}"
            f"  {atoms.a[1, 2] / atoms.s[1]:.6f}\n"
            f"{atoms.s[2]}  {atoms.a[2, 0] / atoms.s[2]:.6f}  {atoms.a[2, 1] / atoms.s[2]:.6f}"
            f"  {atoms.a[2, 2] / atoms.s[2]:.6f}\n"
        )
        # Atomic number (int), atomic charge (float), and atom position (floats) for every atom
        fp.writelines(
            f"{SYMBOL2NUMBER[atoms.atom[ia]]}  {atoms.Z[ia]:.3f}  "
            f"{atoms.pos[ia, 0]: .6f}  {atoms.pos[ia, 1]: .6f}  {atoms.pos[ia, 2]: .6f}\n"
            for ia in range(atoms.Natoms)
        )
        if fods is not None:
            for s in range(len(fods)):
                fp.writelines(
                    f"{SYMBOL2NUMBER[elec_symbols[s]]}  0.000  "
                    f"{ie[0]: .6f}  {ie[1]: .6f}  {ie[2]: .6f}\n"
                    for ie in fods[s]
                )
        # Field data (float) with scientific formatting
        # We have s[0]*s[1] chunks values with a length of s[2]
        for i in range(atoms.s[0] * atoms.s[1]):
            # Print every round of values, so we can add empty lines between them
            data_str = "%+1.6e  " * atoms.s[2] % tuple(field[i * atoms.s[2] : (i + 1) * atoms.s[2]])
            # Print a maximum of 6 values per row
            # Max width for this formatting is 90, since 6*len("+1.00000e-000  ")=90
            # Setting break_on_hyphens to False greatly improves the textwrap.fill performance
            fp.write(f"{textwrap.fill(data_str, width=90, break_on_hyphens=False)}\n\n")
