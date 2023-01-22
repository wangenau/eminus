#!/usr/bin/env python3
'''PDB file handling.'''
import numpy as np

from ..logger import log
from ..units import bohr2ang


def write_pdb(object, filename, fods=None, elec_symbols=None):
    '''Generate pdb files from atoms objects.

    See :func:`~eminus.io.pdb.create_pdb_str` for more information about the pdb file format.

    Args:
        object: Atoms or SCF object.
        filename (str): pdb output file path/name.

    Keyword Args:
        fods (list): FOD coordinates to write.
        elec_symbols (list): Identifier for up and down FODs.

    Returns:
        None.
    '''
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object

    if not filename.endswith('.pdb'):
        filename += '.pdb'

    if elec_symbols is None:
        elec_symbols = ['X', 'He']
        if 'He' in atoms.atom and atoms.Nspin == 2:
            log.warning('You need to modify "elec_symbols" to write helium with FODs in the spin-'
                        'polarized case.')

    atom = atoms.atom
    X = atoms.X
    if fods is not None:
        atom += [elec_symbols[0]] * len(fods[0]) + [elec_symbols[1]] * len(fods[1])
        if fods[0].shape[0] != 0:
            X = np.vstack((X, fods[0]))
        if fods[1].shape[0] != 0:
            X = np.vstack((X, fods[1]))

    with open(filename, 'w') as fp:
        fp.write(create_pdb_str(atom, X, a=atoms.a))
    return


def create_pdb_str(atom, X, a=None):
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
        pdb += '\n'

    # Create molecule data
    pdb += 'MODEL 1'
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
        pdb += f'{X[ia, 0]:8.3f}'.rjust(8)  # 31-38 X orthogonal coordinate
        pdb += f'{X[ia, 1]:8.3f}'.rjust(8)  # 39-46 Y orthogonal coordinate
        pdb += f'{X[ia, 2]:8.3f}'.rjust(8)  # 47-54 Z orthogonal coordinate
        pdb += f'{1:6.2f}'.rjust(6)         # 55-60 Occupancy
        pdb += f'{0:6.2f}'.rjust(6)         # 61-66 Temperature factor
        pdb += '          '
        pdb += f'{atom[ia]}'.rjust(2)       # 77-78 Element symbol
        # pdb += '  '                       # 79-80 Charge
    return f'{pdb}\nENDMDL'
