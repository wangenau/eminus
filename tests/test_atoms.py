#!/usr/bin/env python3
"""Test the Atoms class."""
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from eminus import Atoms, log
from eminus.tools import center_of_mass

inp = ('He', (0, 0, 0))


@pytest.mark.parametrize(('atom', 'ref', 'Nref'), [('H', ['H'], 1),
                                                   (['H'], ['H'], 1),
                                                   ('He-q2', ['He'], 1),
                                                   (['H', 'He'], ['H', 'He'], 2),
                                                   ('CH4', ['C', 'H', 'H', 'H', 'H'], 5),
                                                   ('HeX', ['He', 'X'], 2)])
def test_atom(atom, ref, Nref):
    """Test initialization of the atom variable."""
    atoms = Atoms(atom, [[0, 0, 0]] * Nref)
    assert atoms.atom == ref
    assert atoms.Natoms == Nref


@pytest.mark.parametrize(('X', 'center', 'ref'), [
    ([[0, 0, 0]], None, [(0, 0, 0)]),
    ([[0, 0, 0]], True, [(10, 10, 10)]),
    ([[0, 0, 0]], 'shift', [(10, 10, 10)]),
    ([[0] * 3, [1] * 3], 'rotate', [(0, 0, 0), (np.sqrt(3), 0, 0)]),
    ([[0] * 3, [1] * 3], 'shift', [[9.5] * 3, [10.5] * 3]),
    ([[0] * 3, [1] * 3], True, [[10 - np.sqrt(3) / 2, 10, 10], [10 + np.sqrt(3) / 2, 10, 10]])])  # ???
def test_coordinates(X, center, ref):
    """Test the setting of the atom coordinates."""
    atoms = Atoms(['H'] * len(X), X=X, center=center)
    assert_allclose(np.abs(atoms.X), ref, atol=1e-15)


@pytest.mark.parametrize(('atom', 'Z', 'ref'), [('H', None, [1]),
                                                ('Li', 'pade', [1]),
                                                ('Li', 'pbe', [3]),
                                                ('Li-q3', None, [3]),
                                                ('He2', 2, [2, 2]),
                                                ('CH4', {'C': 3, 'H': 2}, [3, 2, 2, 2, 2])])
def test_charge(atom, Z, ref):
    """Test setting of charges."""
    atoms = Atoms(atom, [[0, 0, 0]] * len(ref))
    atoms.Z = Z
    assert_equal(atoms.Z, ref)


@pytest.mark.parametrize(('s', 'ref'), [(2, [2] * 3),
                                        ([2, 3, 4], [2, 3, 4]),
                                        (None, 99)])
def test_sampling(s, ref):
    """Test the initialization of sampling."""
    atoms = Atoms(*inp)
    if s is not None:
        atoms.s = s
    assert_allclose(atoms.s, ref)


@pytest.mark.parametrize('size', [3, (5, 10, 15)])
def test_cell(size):
    """Test the setting of the cell sampling."""
    atoms = Atoms(*inp, a=size).build()
    assert_allclose(np.diag(atoms.R), size)
    assert_allclose(atoms.Omega, np.prod(atoms.a))
    assert_allclose(atoms.r[0], 0)
    assert len(atoms.r) == atoms.Ns
    assert_allclose(atoms.s[0] / atoms.s[1], atoms.a[0] / atoms.a[1], atol=0.02)


def test_G():
    """Test the setting of G-vectors."""
    atoms = Atoms(*inp)
    atoms.s = 2
    atoms.build()
    assert_allclose(atoms.G[0], 0)
    assert len(atoms.G) == atoms.Ns
    assert_allclose(atoms.G2, atoms.G2c)
    assert_allclose(atoms.Sf, 1)
    atoms = Atoms(*inp)
    atoms.s = 2
    atoms.build()
    assert len(atoms.G2) == len(atoms.G2c)


@pytest.mark.parametrize(('atom', 'unrestricted', 'ref'), [('H', None, 2),
                                                           ('He', None, 1),
                                                           ('H', True, 2),
                                                           ('He', False, 1)])
def test_spin(atom, unrestricted, ref):
    """Test the spin option."""
    atoms = Atoms(atom=atom, X=(0, 0, 0), unrestricted=unrestricted)
    assert atoms.occ.Nspin == ref


@pytest.mark.parametrize(('f', 'unrestricted', 'fref', 'Nref'), [
    (None, False, [[2]], 1),
    (None, True, [[1], [1]], 1),
    (None, False, [[2]], 1),
    (None, True, [[1], [1]], 1),
    (2, False, [[2]], 1),
    (1, True, [[1], [1]], 1),
    (2, False, [[2]], 1),
    (1, True, [[1], [1]], 1)])
def test_occupations(f, unrestricted, fref, Nref):
    """Test the occupation and state options."""
    atoms = Atoms(*inp, unrestricted=unrestricted)
    atoms.s = 1
    atoms.f = f
    atoms.build()
    assert_equal(atoms.f, fref)
    assert atoms.occ.Nstate == Nref


def test_verbose():
    """Test the verbosity level."""
    log.verbose = 'DEBUG'
    atoms = Atoms(*inp)
    assert atoms.verbose == log.verbose
    level = 'DEBUG'
    log.verbose = level
    atoms = Atoms(*inp)
    assert atoms.verbose == level
    atoms = Atoms(*inp, verbose=level)
    assert atoms.verbose == level


def test_operators():
    """Test that operators are properly set and callable."""
    atoms = Atoms(*inp).build()
    assert atoms._base_operator() is None
    for op in ('O', 'L', 'Linv', 'I', 'J', 'Idag', 'Jdag', 'K', 'T'):
        assert callable(getattr(atoms, op))


def test_clear():
    """Test the clear function."""
    atoms = Atoms(*inp).build()
    assert atoms.is_built
    atoms.clear()
    assert not atoms.is_built
    assert [x for x in (atoms.r, atoms.G, atoms.G2, atoms.active, atoms.G2c, atoms.Sf) if x is None]


def test_recenter():
    """Test the recenter function."""
    atoms = Atoms(*inp)
    atoms.s = 2
    atoms.build()
    Sf_old = atoms.Sf
    center = (1, 1, 1)
    atoms.recenter(center)
    assert_allclose(center_of_mass(atoms.X), center)
    assert not np.array_equal(Sf_old, atoms.Sf)
    assert atoms.center == 'recentered'


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
