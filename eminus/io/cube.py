#!/usr/bin/env python3
"""CUBE file handling."""
import textwrap
import time

import numpy as np

from ..data import NUMBER2SYMBOL, SYMBOL2NUMBER
from ..logger import log
from ..version import __version__


def read_cube(filename):
    """Load atom and cell data from cube files.

    There is no standard for cube files. The following format has been used.
    File format definition: https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html

    Args:
        filename (str): cube input file path/name.

    Returns:
        tuple[list, ndarray, float, ndarray, int, ndarray]:
        Species, positions, charges, cell size, sampling, and field array.
    """
    if not filename.endswith('.cube'):
        filename += '.cube'

    # Atomic units and a cuboidal cell that starts at (0,0,0) are assumed.
    with open(filename, 'r') as fh:
        lines = fh.readlines()

        # The first and second line can contain comments, print them if available
        comment = f'{lines[0].strip()}\n{lines[1].strip()}'
        if comment:
            log.info(f'CUBE file comment: "{comment}"')

        # Line 4 to 6 contain the sampling per axis, and the cell basis vectors with length a/s
        # A cuboidal cell is assumed, so only use the diagonal entries
        s = np.empty(3, dtype=int)
        a = np.empty(3)
        for i, line in enumerate(lines[3:6]):
            line_split = line.strip().split()
            s[i] = line_split[0]
            a[i] = s[i] * np.float_(line_split[i + 1])

        atom = []
        X = []
        Z = []
        # Following lines contain atom positions with the format: atom-id charge x-pos y-pos z-pos
        offset = 0
        for line in lines[6:]:
            line_split = line.strip().split()
            # If the first value is not a (positive) integer, we have reached the field data
            if not line_split[0].isdigit():
                break
            atom.append(NUMBER2SYMBOL[int(line_split[0])])
            Z.append(float(line_split[1]))
            X.append(np.float_(line_split[2:5]))
            offset += 1
    X = np.asarray(X)

    # The rest of the data is the field data
    # Split the strings, flatten the lists of lists, and convert to a float numpy array
    field_list = [l.split() for l in lines[6 + offset:]]
    field_list = [item for sublist in field_list for item in sublist]
    field = np.asarray(field_list, dtype=float)
    return atom, X, Z, a, s, field


def write_cube(object, filename, field, fods=None, elec_symbols=None):
    """Generate cube files from given field data.

    There is no standard for cube files. The following format has been used to work with VESTA.
    File format definition: https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html

    Args:
        object: Atoms or SCF object.
        filename (str): cube output file path/name.
        field (ndarray): Real-space field data.

    Keyword Args:
        fods (list): FOD coordinates to write.
        elec_symbols (list): Identifier for up and down FODs.

    Returns:
        None.
    """
    # Atomic units are assumed, so there is no need for conversion.
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object

    if not filename.endswith('.cube'):
        filename += '.cube'

    if elec_symbols is None:
        elec_symbols = ('X', 'He')
        if 'He' in atoms.atom and atoms.Nspin == 2:
            log.warning('You need to modify "elec_symbols" to write helium with FODs in the spin-'
                        'polarized case.')

    # Make sure we have real valued data in the correct order
    field = np.real(field)

    with open(filename, 'w') as fp:
        # The first two lines have to be a comment.
        # Print information about the file and program, and the file creation time.
        fp.write(f'File generated with eminus {__version__} on {time.ctime()}\n\n')
        # Number of atoms (int), and origin of the coordinate system (float)
        # The origin is normally at 0,0,0 but we could move our box, so take the minimum
        if fods is None:
            fp.write(f'{atoms.Natoms}  ')
        else:
            fp.write(f'{atoms.Natoms + sum([len(i) for i in fods])}  ')
        fp.write('0.0  0.0  0.0\n')
        # Number of points per axis (int), and vector defining the axis (float)
        # We only have a cuboidal box, so each vector only has one non-zero component
        fp.write(f'{atoms.s[0]}  {atoms.a[0] / atoms.s[0]:.6f}  0.0  0.0\n'
                 f'{atoms.s[1]}  0.0  {atoms.a[1] / atoms.s[1]:.6f}  0.0\n'
                 f'{atoms.s[2]}  0.0  0.0  {atoms.a[2] / atoms.s[2]:.6f}\n')
        # Atomic number (int), atomic charge (float), and atom position (floats) for every atom
        for ia in range(atoms.Natoms):
            fp.write(f'{SYMBOL2NUMBER[atoms.atom[ia]]}  {atoms.Z[ia]:.3f}  '
                     f'{atoms.X[ia, 0]: .6f}  {atoms.X[ia, 1]: .6f}  {atoms.X[ia, 2]: .6f}\n')
        if fods is not None:
            for s in range(len(fods)):
                for ie in fods[s]:
                    fp.write(f'{SYMBOL2NUMBER[elec_symbols[s]]}  0.000  '
                             f'{ie[0]: .6f}  {ie[1]: .6f}  {ie[2]: .6f}\n')
        # Field data (float) with scientific formatting
        # We have s[0]*s[1] chunks values with a length of s[2]
        for i in range(atoms.s[0] * atoms.s[1]):
            # Print every round of values, so we can add empty lines between them
            data_str = '%+1.6e  ' * atoms.s[2] % tuple(field[i * atoms.s[2]:(i + 1) * atoms.s[2]])
            # Print a maximum of 6 values per row
            # Max width for this formatting is 90, since 6*len('+1.00000e-000  ')=90
            fp.write(f'{textwrap.fill(data_str, width=90)}\n\n')
    return
