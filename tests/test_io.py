#!/usr/bin/env python3
"""Test input and output functionalities."""
import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import (
    Atoms,
    read,
    read_cube,
    read_json,
    read_xyz,
    SCF,
    write,
    write_cube,
    write_json,
    write_pdb,
    write_xyz,
)

atoms = Atoms('LiH', ((0, 0, 0), (3, 0, 0)), s=2, ecut=1)
scf = SCF(atoms)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_xyz(Nspin):
    """Test XYZ file output and input."""
    filename = 'test.xyz'
    fods = [atoms.X] * Nspin
    write(atoms, filename, fods=fods)
    atom, X = read(filename)
    os.remove(filename)
    if Nspin == 1:
        assert atoms.atom + ['X'] * len(atoms.X) == atom
    else:
        assert atoms.atom + ['X'] * len(atoms.X) + ['He'] * len(atoms.X) == atom
    assert_allclose(atoms.X, X[:len(atoms.X)], atol=1e-6)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_cube(Nspin):
    """Test CUBE file output and input."""
    filename = 'test.cube'
    fods = [atoms.X] * Nspin
    write(atoms, filename, scf.W[0, :, 0], fods=fods)
    atom, X, Z, a, s, field = read(filename)
    os.remove(filename)
    if Nspin == 1:
        assert atoms.atom + ['X'] * len(atoms.X) == atom
    else:
        assert atoms.atom + ['X'] * len(atoms.X) + ['He'] * len(atoms.X) == atom
    assert_allclose(atoms.X, X[:len(atoms.X)], atol=1e-6)
    assert_allclose(atoms.Z, Z[:len(atoms.Z)])
    assert_allclose(atoms.a, a)
    assert_allclose(atoms.s, s)
    assert_allclose(np.real(scf.W[0, :, 0]), field, atol=1e-7)


@pytest.mark.parametrize('object', [atoms, scf, scf.energies])
def test_json(object):
    """Test JSON file output and input."""
    filename = 'test.json'
    write(object, filename)
    test = read(filename)
    os.remove(filename)
    for attr in test.__dict__:
        # Skip objects and dictionaries
        if attr in ('atoms', 'GTH', 'log'):
            continue
        try:
            assert_allclose(getattr(object, attr), getattr(test, attr))
        except TypeError:
            assert getattr(object, attr) == getattr(test, attr)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_pdb(Nspin):
    """Just test the PDB output execution, since we have no read function for it."""
    filename = 'test.pdb'
    fods = [atoms.X] * Nspin
    write(atoms, filename, fods=fods)
    os.remove(filename)


@pytest.mark.parametrize('filending', ['pdb', 'xyz'])
def test_trajectory(filending):
    """Test the trajectory keyword that append geometries to a file."""
    filename = f'test.{filending}'
    write(atoms, filename, trajectory=False)
    old_size = os.stat(filename).st_size
    write(atoms, filename, trajectory=True)
    new_size = os.stat(filename).st_size
    os.remove(filename)
    # The trajectory file has to be larger than the original one
    assert old_size < new_size


def test_filename_ending():
    """Test if the functions still work when omiting the filename ending."""
    filename = 'test'
    write_xyz(atoms, filename)
    read_xyz(filename)
    os.remove(filename + '.xyz')
    write_cube(atoms, filename, scf.W[0, :, 0])
    read_cube(filename)
    os.remove(filename + '.cube')
    write_json(atoms, filename)
    read_json(filename)
    os.remove(filename + '.json')
    write_pdb(atoms, filename)
    os.remove(filename + '.pdb')


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
