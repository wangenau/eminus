#!/usr/bin/env python3
"""XYZ file handling."""
import time

import numpy as np

from ..logger import log
from ..units import ang2bohr, bohr2ang
from ..version import __version__


def read_xyz(filename):
    """Load atom species and positions from XYZ files.

    File format definition: https://openbabel.org/wiki/XYZ_%28format%29

    Args:
        filename (str): XYZ input file path/name.

    Returns:
        tuple[list, ndarray]: Atom species and positions.
    """
    if not filename.endswith('.xyz'):
        filename += '.xyz'

    with open(filename, 'r') as fh:
        lines = fh.readlines()

        # The first line contains the number of atoms
        Natoms = int(lines[0].strip())

        # The second line can contain a comment, print it if available
        comment = lines[1].strip()
        if comment:
            log.info(f'XYZ file comment: "{comment}"')

        atom = []
        pos = []
        # Following lines contain atom positions with the format: Atom x-pos y-pos z-pos
        for line in lines[2:2 + Natoms]:
            line_split = line.strip().split()
            atom.append(line_split[0])
            pos.append(np.float_(line_split[1:4]))

    # xyz files are in Angstrom, so convert to Bohr
    pos = ang2bohr(np.asarray(pos))
    return atom, pos


def write_xyz(obj, filename, fods=None, elec_symbols=('X', 'He'), trajectory=False):
    """Generate XYZ files from atoms objects.

    File format definition: https://openbabel.org/wiki/XYZ_%28format%29

    Args:
        obj: Atoms or SCF object.
        filename (str): XYZ output file path/name.

    Keyword Args:
        fods (list): FOD coordinates to write.
        elec_symbols (list): Identifier for up and down FODs.
        trajectory (bool): Allow appending to a file to create trajectories.

    Returns:
        None.
    """
    atoms = obj._atoms

    if not filename.endswith('.xyz'):
        filename += '.xyz'

    # Convert the coordinates from atomic units to Angstrom
    pos = bohr2ang(atoms.pos)
    if fods is not None:
        fods = [bohr2ang(i) for i in fods]

    if 'He' in atoms.atom and atoms.unrestricted:
        log.warning('You need to modify "elec_symbols" to write helium with FODs in the spin-'
                    'polarized case.')

    # Append to a file when using the trajectory keyword
    if trajectory:
        mode = 'a'
    else:
        mode = 'w'

    with open(filename, mode) as fp:
        # The first line contains the number of atoms
        # If we add FOD coordinates, add them to the count
        if fods is None:
            fp.write(f'{atoms.Natoms}\n')
        else:
            fp.write(f'{atoms.Natoms + sum([len(i) for i in fods])}\n')
        # The second line can contain a comment
        # Print information about the file and program, and the file creation time
        fp.write(f'File generated with eminus {__version__} on {time.ctime()}\n')
        for ia in range(atoms.Natoms):
            fp.write(f'{atoms.atom[ia]:<2s}  '
                     f'{pos[ia, 0]: .6f}  {pos[ia, 1]: .6f}  {pos[ia, 2]: .6f}\n')
        # Add FOD coordinates if desired
        # The atom symbol will default to pos (no atom type)
        if fods is not None:
            for s in range(len(fods)):
                for ie in fods[s]:
                    fp.write(f'{elec_symbols[s]:<2s}  {ie[0]: .6f}  {ie[1]: .6f}  {ie[2]: .6f}\n')
