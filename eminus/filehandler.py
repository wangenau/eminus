#!/usr/bin/env python3
'''Import and export functionalities.'''
import glob
import os
import pickle
import textwrap
import time

import numpy as np

from . import __path__
from .data import number2symbol, symbol2number
from .logger import log
from .units import ang2bohr, bohr2ang
from .version import __version__


def read_xyz(filename):
    '''Load atom species and positions from xyz files.

    File format definition: https://openbabel.org/wiki/XYZ_%28format%29

    Args:
        filename (str): xyz input file path/name.

    Returns:
        tuple[list, ndarray]: Atom species and positions.
    '''
    if not filename.endswith('.xyz'):
        filename = f'{filename}.xyz'

    with open(filename, 'r') as fh:
        lines = fh.readlines()

        # The first line contains the number of atoms
        Natoms = int(lines[0].strip())

        # The second line can contain a comment, print it if available
        comment = lines[1].strip()
        log.info(f'XYZ file comment: "{comment}"')

        atom = []
        X = []
        # Following lines contain atom positions with the format: Atom x-pos y-pos z-pos
        for line in lines[2:2 + Natoms]:
            line_split = line.strip().split()
            atom.append(line_split[0])
            X.append(np.float_(line_split[1:4]))

    # xyz files are in Angstrom, so convert to Bohr
    X = ang2bohr(np.asarray(X))
    return atom, X


def write_xyz(object, filename, extra=None):
    '''Generate xyz files from atoms objects.

    File format definition: https://openbabel.org/wiki/XYZ_%28format%29

    Args:
        object: Atoms or SCF object.
        filename (str): xyz output file path/name.

    Keyword Args:
        extra (ndarray): Extra coordinates to write.
    '''
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object
    atom = atoms.atom
    Natoms = atoms.Natoms
    X = atoms.X

    if not filename.endswith('.xyz'):
        filename = f'{filename}.xyz'

    # Convert the coordinates from atomic units to Angstrom
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
        # Print information about the file and program, and the file creation time.
        fp.write(f'File generated with eminus {__version__} on {time.ctime()}\n')
        for ia in range(Natoms):
            fp.write(f'{atom[ia]}  {X[ia][0]:.5f}  {X[ia][1]:.5f}  {X[ia][2]:.5f}\n')
        # Add extra coordinates if desired. The atom symbol will default to X (no atom type).
        if extra is not None:
            for ie in extra:
                fp.write(f'X  {ie[0]:.5f}  {ie[1]:.5f}  {ie[2]:.5f}\n')
    return


def read_cube(filename):
    '''Load atom and cell data from cube files.

    There is no standard for cube files. The following format has been used.
    File format definition: https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html

    Args:
        filename (str): cube input file path/name.

    Returns:
        tuple[list, ndarray, float, ndarray, int]: Species, positions, charges, cell size, sampling.
    '''
    if not filename.endswith('.cube'):
        filename = f'{filename}.cube'

    # Atomic units and a cuboidal unit cell that starts at (0,0,0) are assumed.
    with open(filename, 'r') as fh:
        lines = fh.readlines()

        # The first and second line can contain comments, print them if available
        comment = f'{lines[0].strip()}\n{lines[1].strip()}'
        log.info(f'XYZ file comment: "{comment}"')

        # Line 4 to 6 contain the sampling per axis, and the unit cell basis vectors with length a/s
        # A cuboidal unit cell is assumed, so only use the diagonal entries
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
        for line in lines[6:]:
            line_split = line.strip().split()
            # If the first value is not a (positive) integer, we have reached the field data
            if not line_split[0].isdigit():
                break
            atom.append(number2symbol[int(line_split[0])])
            Z.append(line_split[1])
            X.append(np.float_(line_split[2:5]))

    X = np.asarray(X)
    return atom, X, Z, a, s


def write_cube(object, field, filename, extra=None):
    '''Generate cube files from given field data.

    There is no standard for cube files. The following format has been used to work with VESTA.
    File format definition: https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html

    Args:
        object: Atoms or SCF object.
        field (ndarray): Real-space field data.
        filename (str): xyz output file path/name.

    Keyword Args:
        extra (ndarray): Extra coordinates to write.
    '''
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object
    # Atomic units are assumed, so there is no need for conversion.
    atom = atoms.atom
    Natoms = atoms.Natoms
    a = atoms.a
    r = atoms.r
    s = atoms.s
    X = atoms.X
    Z = atoms.Z

    if not filename.endswith('.cube'):
        filename = f'{filename}.cube'

    # Our field data has been created in a different order than needed for cube files
    # (triple loop over z,y,x instead of x,y,z), so rearrange it with some index magic.
    idx = []
    for Nx in range(s[0]):
        for Ny in range(s[1]):
            for Nz in range(s[2]):
                idx.append(Nx + Ny * s[0] + Nz * s[0] * s[1])
    idx = np.asarray(idx)

    # Make sure we have real valued data in the correct order
    field = np.real(field[idx])

    with open(filename, 'w') as fp:
        # The first two lines have to be a comment.
        # Print information about the file and program, and the file creation time.
        fp.write(f'File generated with eminus {__version__} on {time.ctime()}\n\n')
        # Number of atoms (int), and origin of the coordinate system (float)
        # The origin is normally at 0,0,0 but we could move our box, so take the minimum
        if extra is None:
            fp.write(f'{Natoms}  ')
        else:
            fp.write(f'{Natoms + len(extra)}  ')
        fp.write(f'{min(r[:, 0]):.5f}  {min(r[:, 1]):.5f}  {min(r[:, 2]):.5f}\n')
        # Number of points per axis (int), and vector defining the axis (float)
        # We only have a cuboidal box, so each vector only has one non-zero component
        fp.write(f'{s[0]}  {a[0] / s[0]:.5f}  0.0  0.0\n')
        fp.write(f'{s[1]}  0.0  {a[1] / s[1]:.5f}  0.0\n')
        fp.write(f'{s[2]}  0.0  0.0  {a[2] / s[2]:.5f}\n')
        # Atomic number (int), atomic charge (float), and atom position (floats) for every atom
        for ia in range(Natoms):
            fp.write(f'{symbol2number[atom[ia]]}  {Z[ia]:.5f}  ')
            fp.write(f'{X[ia][0]:.5f}  {X[ia][1]:.5f}  {X[ia][2]:.5f}\n')
        if extra is not None:
            for ie in extra:
                fp.write('0  0.00000  ')
                fp.write(f'{ie[0]:.5f}  {ie[1]:.5f}  {ie[2]:.5f}\n')
        # Field data (float) with scientific formatting
        # We have s[0]*s[1] chunks values with a length of s[2]
        for i in range(s[0] * s[1]):
            # Print every round of values, so we can add empty lines between them
            data_str = '%+1.5e  ' * s[2] % tuple(field[i * s[2]:(i + 1) * s[2]])
            # Print a maximum of 6 values per row
            # Max width for this formatting is 90, since 6*len('+1.00000e-000  ')=90
            fp.write(f'{textwrap.fill(data_str, width=90)}\n\n')
    return


def save(object, filename):
    '''Save objects in a pickle file.

    This function is for personal use only. Never load a file you haven't saved yourself!

    Args:
        object: Class object.
        filename (str): xyz input file path/name.
    '''
    if not filename.endswith(('.pickle', '.pkl')):
        filename = f'{filename}.pickle'

    with open(filename, 'wb') as fp:
        pickle.dump(object, fp, pickle.HIGHEST_PROTOCOL)
    return


def load(filename):
    '''Load objects from a pickle file.

    This function is for personal use only. Never load a file you haven't saved yourself!

    Args:
        filename (str): xyz input file path/name.

    Returns:
        Class object.
    '''
    if not filename.endswith(('.pickle', '.pkl')):
        filename = f'{filename}.pickle'

    with open(filename, 'rb') as fh:
        return pickle.load(fh)


def create_pdb(atom, X, a=None):
    '''Convert atom symbols and positions to the pdb format.

    File format definitions:
        CRYST1: https://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#CRYST1

        ATOM: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM

    Args:
        atom (list): Atom symbols.
        X (ndarray): Atom positions.

    Keyword Args:
        a (float): Cell size.

    Returns:
        str: pdb file format.
    '''
    # Convert Bohr to Angstrom
    X = bohr2ang(X)
    if a is not None:
        a = bohr2ang(a)

    # pdb files have specific numbers of characters for every data with changing justification
    # Write everything explicitly down to not loose track of line lengths
    pdb = ''
    # Create data for a cuboidal cell
    if a is not None:
        pdb += 'CRYST1'                 # 1-6 "CRYST1"
        pdb += f'{a[0]:9.3f}'.rjust(9)  # 7-15 a
        pdb += f'{a[1]:9.3f}'.rjust(9)  # 16-24 b
        pdb += f'{a[2]:9.3f}'.rjust(9)  # 25-33 c
        pdb += f'{90:7.2f}'.rjust(7)    # 34-40 alpha
        pdb += f'{90:7.2f}'.rjust(7)    # 41-47 beta
        pdb += f'{90:7.2f}'.rjust(7)    # 48-54 gamma
        pdb += ' '
        pdb += 'P 1'.ljust(11)          # 56-66 Space group
        # pdb += '1'.rjust(4)           # 67-70 Z value

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


def read_gth(atom, charge=None, psp_path=None):
    '''Read GTH files for a given atom.

    Args:
        atom (str): Atom name.

    Keyword Args:
        charge (int): Valence charge.
        psp_path (str): Path to GTH pseudopotential files. Defaults to installation_path/pade_gth/.

    Returns:
        dict: GTH parameters.
    '''
    if psp_path is None:
        psp_path = f'{__path__[0]}/pade_gth/'

    if charge is not None:
        f_psp = f'{psp_path}{atom}-q{charge}.gth'
    else:
        files = glob.glob(f'{psp_path}{atom}-q*')
        files.sort()
        try:
            f_psp = files[0]
        except IndexError:
            log.exception(f'There is no GTH pseudopotential in {psp_path} for "{atom}"')
            raise
        if len(files) > 1:
            log.info(f'Multiple pseudopotentials found for "{atom}". '
                     f'Continue with "{os.path.basename(f_psp)}".')

    psp = {}
    cloc = np.zeros(4)
    rp = np.zeros(4)
    Nproj_l = np.zeros(4, dtype=int)
    h = np.zeros([4, 3, 3])
    try:
        with open(f_psp, 'r') as fh:
            # Skip the first line, since we don't need the atom symbol here. If needed, use
            # psp['atom'] = fh.readline().split()[0]  # Atom symbol
            fh.readline()
            N_all = fh.readline().split()
            N_s, N_p, N_d, N_f = int(N_all[0]), int(N_all[1]), int(N_all[2]), int(N_all[3])
            psp['Zion'] = N_s + N_p + N_d + N_f  # Ionic charge
            # Skip the number of local coefficients, since we don't need it. If needed, use
            # rloc, n_c_local = fh.readline().split()
            # psp['n_c_local'] = int(n_c_local)  # Number of local coefficients
            rloc = fh.readline().split()[0]
            psp['rloc'] = float(rloc)  # Range of local Gaussian charge distribution
            for i, val in enumerate(fh.readline().split()):
                cloc[i] = float(val)
            psp['cloc'] = cloc  # Coefficients for the local part
            lmax = int(fh.readline().split()[0])
            psp['lmax'] = lmax  # Maximal angular momentum in the non-local part
            for iiter in range(lmax):
                rp[iiter], Nproj_l[iiter] = [float(i) for i in fh.readline().split()]
                for jiter in range(Nproj_l[iiter]):
                    tmp = fh.readline().split()
                    for iread, kiter in enumerate(range(jiter, Nproj_l[iiter])):
                        h[iiter, jiter, kiter] = float(tmp[iread])
            psp['rp'] = rp  # Projector radius for each angular momentum
            psp['Nproj_l'] = Nproj_l  # Number of non-local projectors
            for k in range(3):
                for i in range(2):
                    for j in range(i + 1, 2):
                        h[k, j, i] = h[k, i, j]
            psp['h'] = h  # Projector coupling coefficients per AM channel
    except FileNotFoundError:
        log.exception(f'There is no GTH pseudopotential for "{os.path.basename(f_psp)}"')
        raise
    return psp
