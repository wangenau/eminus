#!/usr/bin/env python3
'''
Import and export functionalities.
'''
from pickle import dump, HIGHEST_PROTOCOL, load
from textwrap import fill
from time import ctime

import numpy as np

from .data import number2symbol, symbol2number
from .units import ang2bohr, bohr2ang
from .version import __version__


def read_xyz(filename, info=False):
    '''Load atom species and positions from xyz files.

    Args:
        filename : str
            xyz input file path/name.

    Kwargs:
        info : bool
            Display file comments.

    Returns:
        Atom species and positions as a tuple(list, array).
    '''
    # XYZ file definitions: https://en.wikipedia.org/wiki/XYZ_file_format
    with open(filename, 'r') as fh:
        lines = fh.readlines()

    # The first line contains the number of atoms
    Natoms = int(lines[0].strip())

    # The second line can contain a comment, print it if available
    comment = lines[1].strip()
    if info:
        print(f'XYZ file comment: "{comment}"')

    atom = []
    X = []
    # Following lines contain atom positions with the format: Atom x-pos y-pos z-pos
    for line in lines[2:2 + Natoms]:
        line_split = line.strip().split()
        atom.append(line_split[0])
        X.append(np.float_(line_split[1:4]))

    # xyz files are in angstrom, so convert to bohr
    X = ang2bohr(np.asarray(X))
    return atom, X


def write_xyz(atoms, filename, extra=None):
    '''Generate xyz files from atoms objects.

    Args:
        atoms :
            Atoms object.

        filename : str
            xyz output file path/name.

    Kwargs:
        extra : array
            Extra coordinates to write.
    '''
    # XYZ file definitions: https://en.wikipedia.org/wiki/XYZ_file_format
    atom = atoms.atom
    Natoms = atoms.Natoms
    X = atoms.X

    if not filename.endswith('.xyz'):
        filename = f'{filename}.xyz'

    # Convert the coordinates from atomic units to angstrom
    X = bohr2ang(X)
    if extra is not None:
        extra = bohr2ang(extra)

    with open(filename, 'w') as fp:
        # The first line contains the number of atoms.
        # If we add extra coordinates, add them to the count.
        if extra is None:
            fp.write(f'{Natoms}\n')
        else:
            fp.write(f'{Natoms + len(extra)}\n')
        # The second line can contains a comment.
        # Print informations about the file and program, and the file creation time.
        fp.write(f'XYZ file generated with eminus {__version__} at {ctime()}\n')
        for ia in range(Natoms):
            fp.write(f'{atom[ia]}  {X[ia][0]:.5f}  {X[ia][1]:.5f}  {X[ia][2]:.5f}\n')
        # Add extra coordinates, if desired. The will default to X (no atom type).
        if extra is not None:
            for ie in extra:
                fp.write(f'X  {ie[0]:.5f}  {ie[1]:.5f}  {ie[2]:.5f}\n')
    return


def read_cube(filename, info=False):
    '''Load atom and cell data from cube files.

    Args:
        filename : str
            cube input file path/name.
    Kwargs:
        info : bool
            Display file comments.

    Returns:
        Species, positions, charges, cell size, and sampling as a
        tuple(list, array, float, array, int).
    '''
    # It seems, that there is no standard for cube files. The following definition is taken from:
    # https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html
    # Atomic units and a cuboidic unit cell that starts at (0,0,0) is assumed.
    with open(filename, 'r') as fh:
        lines = fh.readlines()

    # The first and second line can contain comments, print them if available
    comment = f'{lines[0].strip()}\n{lines[1].strip()}'
    if info:
        print(f'XYZ file comment: "{comment}"')

    # Line 4 to 6 contain the sampling per axis, and the unit cell basis vectors with length a/S
    # A cuboidic unit cell is assumed, so only use the diagonal entries
    S = np.empty(3)
    a = np.empty(3)
    for i, line in enumerate(lines[3:6]):
        line_split = line.strip().split()
        S[i] = int(line_split[0])
        a[i] = S[i] * np.float_(line_split[i + 1])

    atom = []
    X = []
    Z = []
    # Following lines contain atom positions with the format: atom-id charge x-pos y-pos z-pos
    for line in lines[6:]:
        line_split = line.strip().split()
        # If the first value is not a (positive) integer, we have reached the field data
        if not line_split[0].isdigit():
            break
        atom.append(number2symbol[int(line_split[0])])
        Z.append(line_split[1])
        X.append(np.float_(line_split[2:5]))

    X = np.asarray(X)
    return atom, X, Z, a, S


def write_cube(atoms, field, filename, extra=None):
    '''Generate cube files from given field data.

    Args:
        atoms :
            Atoms object.

        field : array
            Real-space field data.

        filename : str
            xyz output file path/name.

    Kwargs:
        extra : array
            Extra coordinates to write.
    '''
    # It seems, that there is no standard for cube files. The following definition will work with
    # VESTA and is taken from: https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html
    # Atomic units are assumed, so there is no need for conversion.
    atom = atoms.atom
    Natoms = atoms.Natoms
    a = atoms.a
    r = atoms.r
    S = atoms.S
    X = atoms.X
    Z = atoms.Z

    if not filename.endswith('.cube'):
        filename = f'{filename}.cube'

    # Our field data has been created in a different order than needed for cube files
    # (triple loop over z,y,x instead of x,y,z), so rearrange it with some index magic.
    idx = []
    for Nx in range(S[0]):
        for Ny in range(S[1]):
            for Nz in range(S[2]):
                idx.append(Nx + Ny * S[0] + Nz * S[0] * S[1])
    idx = np.asarray(idx)

    # Make sure we have real valued data in the correct order
    field = np.real(field[idx])

    with open(filename, 'w') as fp:
        # The first two lines have to be a comment.
        # Print file creation time and informations about the file and program.
        fp.write(f'{ctime()}\n')
        fp.write(f'Cube file generated with eminus {__version__}\n')
        # Number of atoms (int), and origin of the coordinate system (float)
        # The origin is normally at 0,0,0 but we could move our box, so take the minimum
        if extra is None:
            fp.write(f'{Natoms}  ')
        else:
            fp.write(f'{Natoms + len(extra)}  ')
        fp.write(f'{min(r[:, 0]):.5f}  {min(r[:, 1]):.5f}  {min(r[:, 2]):.5f}\n')
        # Number of points per axis (int), and vector defining the axis (float)
        # We only have a cuboidic box, so each vector only has one non-zero component
        fp.write(f'{S[0]}  {a[0] / S[0]:.5f}  0.0  0.0\n')
        fp.write(f'{S[1]}  0.0  {a[1] / S[1]:.5f}  0.0\n')
        fp.write(f'{S[2]}  0.0  0.0  {a[2] / S[2]:.5f}\n')
        # Atomic number (int), atomic charge (float), and atom position (floats) for every atom
        for ia in range(Natoms):
            fp.write(f'{symbol2number[atom[ia]]}  {Z[ia]:.5f}  ')
            fp.write(f'{X[ia][0]:.5f}  {X[ia][1]:.5f}  {X[ia][2]:.5f}\n')
        if extra is not None:
            for ie in extra:
                fp.write('0  0.00000  ')
                fp.write(f'{ie[0]:.5f}  {ie[1]:.5f}  {ie[2]:.5f}\n')
        # Field data (float) with scientific formatting
        # We have S[0]*S[1] chunks values with a length of S[2]
        for i in range(S[0] * S[1]):
            # Print every round of values, so we can add empty lines between them
            data_str = '%+1.5e  ' * S[2] % tuple(field[i * S[2]:(i + 1) * S[2]])
            # Print a maximum of 6 values per row
            # Max width for this formatting is 90, since 6*len('+1.00000e-000  ')=90
            fp.write(f'{fill(data_str, width=90)}\n\n')
    return


def save_atoms(atoms, filename):
    '''Save atoms objects to a pickle file.

    Args:
        atoms :
            Atoms object.

        filename : str
            xyz input file path/name.
    '''
    with open(filename, 'wb') as fp:
        dump(atoms, fp, HIGHEST_PROTOCOL)
    return


def load_atoms(filename):
    '''Load atoms objects from a pickle file.

    Args:
        filename : str
            xyz input file path/name.

    Returns:
        Atoms object.
    '''
    with open(filename, 'rb') as fh:
        return load(fh)


def create_pdb(atom, X, a=None):
    '''Convert xyz files to pdb format.

    Args:
        atom : list
            Atom symbols.

        X : array
            Atom positions.

    Kwargs:
        a : float
            Cell size.

    Returns:
        pdb file format as a string.
    '''
    # pdb file definitions:
    # For CRYST1: https://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#CRYST1
    # For ATOM: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
    # Convert Bohr to Angstrom
    X = bohr2ang(X)
    if a is not None:
        a = bohr2ang(a)

    # pdb files have specific numbers of characters for every data with changing justification
    # Write everything explicitly down to not loose track of line lengths
    pdb = ''
    # Create data for a cuboidic cell
    if a is not None:
        pdb += 'CRYST1'               # 1-6 "CRYST1"
        pdb += f'{a[0]:9.3f}'.rjust(9)   # 7-15 a
        pdb += f'{a[1]:9.3f}'.rjust(9)   # 16-24 b
        pdb += f'{a[2]:9.3f}'.rjust(9)   # 25-33 c
        pdb += f'{90:7.2f}'.rjust(7)  # 34-40 alpha
        pdb += f'{90:7.2f}'.rjust(7)  # 41-47 beta
        pdb += f'{90:7.2f}'.rjust(7)  # 48-54 gamma
        pdb += ' '
        pdb += 'P 1'.ljust(11)        # 56-66 Space group
        # pdb += '1'.rjust(4)         # 67-70 Z value

    # Create molecule data
    pdb += '\nMODEL 1'
    for ia in range(len(atom)):
        pdb += '\nATOM  '                   # 1-6 "ATOM"
        pdb += f'{ia + 1}'.rjust(5)         # 7-11 Atom serial number
        pdb += ' '
        pdb += f'{atom[ia]}'.rjust(4)       # 13-16 Atom name
        pdb += ' '                          # 17 Alternate location indicator
        pdb += 'MOL'                        # 18-20 Residue name
        pdb += ' '
        pdb += ' '                          # 22 Chain identifier
        pdb += '1'.rjust(4)                 # 23-26 Residue sequence number
        pdb += ' '                          # 27 Code for insertions of residues
        pdb += '   '
        pdb += f'{X[ia][0]:8.3f}'.rjust(8)  # 31-38 X orthogonal coordinate
        pdb += f'{X[ia][1]:8.3f}'.rjust(8)  # 39-46 Y orthogonal coordinate
        pdb += f'{X[ia][2]:8.3f}'.rjust(8)  # 47-54 Z orthogonal coordinate
        pdb += f'{1:6.2f}'.rjust(6)         # 55-60 Occupancy
        pdb += f'{0:6.2f}'.rjust(6)         # 61-66 Temperature factor
        pdb += '          '
        pdb += f'{atom[ia]}'.rjust(2)       # 77-78 Element symbol
        # pdb += '  '                       # 79-80 Charge
    pdb = f'{pdb}\nENDMDL'
    return pdb
