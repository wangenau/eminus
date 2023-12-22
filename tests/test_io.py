#!/usr/bin/env python3
"""Test input and output functionalities."""
import copy
import os

from numpy.testing import assert_allclose, assert_equal
import pytest

from eminus import (
    Atoms,
    read,
    read_cube,
    read_json,
    read_traj,
    read_xyz,
    SCF,
    write,
    write_cube,
    write_json,
    write_pdb,
    write_traj,
    write_xyz,
)

atoms = Atoms('LiH', ((0, 0, 0), (3, 0, 0)), ecut=1)
scf = SCF(atoms, opt={'sd': 1})
scf.run()


@pytest.mark.parametrize('Nspin', [1, 2])
def test_xyz(Nspin):
    """Test XYZ file output and input."""
    filename = 'test.xyz'
    fods = [atoms.pos] * Nspin
    write(atoms, filename, fods=fods)
    atom, pos = read(filename)
    os.remove(filename)
    if Nspin == 1:
        assert atoms.atom + ['X'] * atoms.Natoms == atom
    else:
        assert atoms.atom + ['X'] * atoms.Natoms + ['He'] * atoms.Natoms == atom
    assert_allclose(atoms.pos, pos[:atoms.Natoms], atol=1e-6)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_cube(Nspin):
    """Test CUBE file output and input."""
    filename = 'test.cube'
    fods = [atoms.pos] * Nspin
    write(atoms, filename, scf.n, fods=fods)
    atom, pos, Z, a, s, field = read(filename)
    os.remove(filename)
    if Nspin == 1:
        assert atoms.atom + ['X'] * atoms.Natoms == atom
    else:
        assert atoms.atom + ['X'] * atoms.Natoms + ['He'] * atoms.Natoms == atom
    assert_allclose(atoms.pos, pos[:atoms.Natoms], atol=1e-6)
    assert_equal(atoms.Z, Z[:atoms.Natoms])
    assert_equal(atoms.a, a)
    assert_equal(atoms.s, s)
    assert_allclose(scf.n, field, atol=1e-9)


def test_cube_noncubic():
    """Test CUBE file output and input for a non-cubic unit cell."""
    filename = 'test.cube'
    atoms = Atoms('LiH', ((0, 0, 0), (3, 0, 0)), ecut=1)
    atoms.a = [[0, 5, 5], [5, 0, 5], [5, 5, 0]]
    scf = SCF(atoms, opt={'sd': 1})
    scf.run()
    write(atoms, filename, scf.n)
    atom, pos, Z, a, s, field = read(filename)
    os.remove(filename)
    assert_allclose(atoms.a, a, atol=2e-6)
    assert_equal(atoms.s, s)
    assert_allclose(scf.n, field, atol=1e-8)


@pytest.mark.parametrize('obj', [atoms, atoms.occ, scf, scf.energies, scf.gth])
def test_json(obj):
    """Test JSON file output and input."""
    filename = 'test.json'
    write(obj, filename)
    test = read(filename)
    os.remove(filename)
    for attr in test.__dict__:
        # Skip objects and dictionaries
        if attr in {'_atoms', 'gth', 'log', '_precomputed'}:
            continue
        if attr == 'GTH':
            assert getattr(obj, attr).keys() == getattr(test, attr).keys()
            continue
        try:
            assert_allclose(getattr(obj, attr), getattr(test, attr))
        except TypeError:
            assert getattr(obj, attr) == getattr(test, attr)


def test_json_restart():
    """Test the SCF restart from JSON files."""
    filename = 'test.json'
    write(scf, filename)
    test_scf = read(filename)
    test_scf.run()
    os.remove(filename)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_pdb(Nspin):
    """Just test the PDB output execution, since we have no read function for it."""
    filename = 'test.pdb'
    fods = [atoms.pos] * Nspin
    write(atoms, filename, fods=fods)
    os.remove(filename)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_traj(Nspin):
    """Test TRAJ file output and input."""
    filename = 'test.traj'
    fods = [atoms.pos] * Nspin
    write(atoms, filename, fods=fods)
    trajectory = read(filename)
    os.remove(filename)
    if Nspin == 1:
        assert atoms.atom + ['X'] * atoms.Natoms == trajectory[0][0]
    else:
        assert atoms.atom + ['X'] * atoms.Natoms + ['He'] * atoms.Natoms == trajectory[0][0]
    assert_allclose(atoms.pos, trajectory[0][1][:atoms.Natoms], atol=1e-6)

    atoms2 = copy.deepcopy(atoms)
    atoms2.pos += 1
    write([atoms, atoms2], filename, fods=fods)
    trajectory = read(filename)
    os.remove(filename)
    if Nspin == 1:
        assert atoms.atom + ['X'] * atoms.Natoms == trajectory[0][0]
        assert atoms2.atom + ['X'] * atoms2.Natoms == trajectory[1][0]
    else:
        assert atoms.atom + ['X'] * atoms.Natoms + ['He'] * atoms.Natoms == trajectory[0][0]
        assert atoms2.atom + ['X'] * atoms2.Natoms + ['He'] * atoms2.Natoms == trajectory[1][0]
    assert_allclose(atoms.pos, trajectory[0][1][:atoms.Natoms], atol=1e-6)
    assert_allclose(atoms2.pos, trajectory[1][1][:atoms2.Natoms], atol=1e-6)


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
    write_cube(atoms, filename, scf.n)
    read_cube(filename)
    os.remove(filename + '.cube')
    write_json(atoms, filename)
    read_json(filename)
    os.remove(filename + '.json')
    write_pdb(atoms, filename)
    os.remove(filename + '.pdb')
    write_traj(atoms, filename)
    read_traj(filename)
    os.remove(filename + '.traj')


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
