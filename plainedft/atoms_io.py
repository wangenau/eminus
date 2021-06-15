#!/usr/bin/env python3
'''
Import and export functionality for Atoms objects.
'''
from textwrap import fill
from time import ctime
from pickle import dump, load, HIGHEST_PROTOCOL
import numpy as np
import plainedft


def save_atoms(atoms, filename):
    '''Save atoms objects into a pickle files.'''
    # TODO: Add remove member functionality
    with open(filename, 'wb') as output:
        dump(atoms, output, HIGHEST_PROTOCOL)
    return


def load_atoms(filename):
    '''Load atoms objects from pickle files.'''
    with open(filename, 'rb') as input:
        return load(input)


def write_cube(atoms, field, filename):
    '''Generate cube files from a given (real-space) field.'''
    # It seems, that there is no standard for cube files. The following definition will work with
    # VESTA and is taken from: https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html
    # Atomic units are assumed, so there is no need for conversion.
    a = atoms.a        # Unit cell size
    r = atoms.r        # Real space sampling points
    S = atoms.S        # Sampling per direction
    X = atoms.X        # Atom positions
    Z = atoms.Z        # Charges

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

    with open(filename, 'w') as f:
        # The first two lines have to be a comment.
        # Print file creation time and informations about the file and program.
        f.write(f'{ctime()}\n')
        f.write(f'Cube file generated with PlaineDFT {plainedft.__version__}\n')
        # Number of atoms (int), and origin of the coordinate system (float)
        # The origin is normally at 0,0,0 but we could move our box, so take the minimum
        f.write(f'{len(X)}  {min(r[:, 0]):.5f}  {min(r[:, 1]):.5f}  {min(r[:, 2]):.5f}\n')
        # Number of points per axis (int), and vector defining the axis (float)
        # We only have a cubic box, so each vector only has one non-zero component
        f.write(f'{S[0]}  {a / S[0]:.5f}  0.0  0.0\n')
        f.write(f'{S[1]}  0.0  {a / S[1]:.5f}  0.0\n')
        f.write(f'{S[2]}  0.0  0.0  {a / S[2]:.5f}\n')
        # Atomic number (int), atomic charge (float), and atom position (floats) for every atom
        # FIXME: Atomic charge can differ from atomic number when only treating valence electrons
        for ia in range(len(X)):
            f.write(f'{Z[ia]}  {Z[ia]:.5f}  {X[ia][0]:.5f}  {X[ia][1]:.5f}  {X[ia][2]:.5f}\n')
        # Field data (float) with scientific formatting
        # We have S[0]*S[1] chunks values with a length of S[2]
        for i in range(S[0] * S[1]):
            # Print every round of values, so we can add empty lines between them
            data_str = '%+1.5e  ' * S[2] % tuple(field[i * S[2]:(i + 1) * S[2]])
            # Print a maximum of 6 values per row
            # Max width for this formatting is 90, since 6*len('+1.00000e-000  ')=90
            f.write(fill(data_str, width=90) + '\n\n')
    return
