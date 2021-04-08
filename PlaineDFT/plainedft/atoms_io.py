#!/usr/bin/env python3
'''
Import and export functionality for Atoms objects.
'''
from pickle import dump, load, HIGHEST_PROTOCOL


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
