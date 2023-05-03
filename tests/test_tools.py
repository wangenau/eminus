#!/usr/bin/env python3
'''Test tools functions.'''
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, SCF
from eminus.dft import get_psi
from eminus.tools import (center_of_mass, check_norm, check_ortho, check_orthonorm,
                          cutoff2gridspacing, get_dipole, get_ip, get_isovalue, gridspacing2cutoff,
                          inertia_tensor, orbital_center)


atoms = Atoms('He2', ((0, 0, 0), (10, 0, 0)), s=15, Nspin=1, center=True)
scf = SCF(atoms)
scf.run()
psi = atoms.I(get_psi(scf, scf.W))


def test_cutoff_and_gridspacing():
    '''Test the cutoff and grid spacing conversion.'''
    rng = default_rng()
    test = np.abs(rng.random(100))
    out = cutoff2gridspacing(gridspacing2cutoff(test))
    assert_allclose(out, test)


@pytest.mark.parametrize('coords, masses, ref', [(np.eye(3), None, [1 / 3] * 3),
                                                 (np.eye(3), np.arange(3), [0, 1 / 3, 2 / 3])])
def test_center_of_mass(coords, masses, ref):
    '''Test the center of mass calculation.'''
    out = center_of_mass(coords, masses)
    assert_allclose(out, ref)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_pycom(Nspin):
    '''Test PyCOM routine.'''
    atoms = Atoms('He2', ((0, 0, 0), (10, 0, 0)), s=10, Nspin=Nspin, center=True).build()
    scf = SCF(atoms)
    scf.run()
    psi = atoms.I(get_psi(scf, scf.W))
    for spin in range(Nspin):
        assert_allclose(orbital_center(atoms, psi)[spin], [[10] * 3] * 2, atol=1e-1)


@pytest.mark.parametrize('coords', [np.array([[1, 0, 0]]),
                                    np.array([[0, 1, 0]]),
                                    np.array([[0, 0, 1]])])
def test_inertia_tensor(coords):
    '''Test the inertia tensor calculation.'''
    out = inertia_tensor(coords)
    ref = np.eye(3)
    ref[np.nonzero(coords[0])] = [0] * 3
    assert_allclose(out, ref)


def test_get_dipole():
    '''Test the electric dipole moment calculation.'''
    assert_allclose(get_dipole(scf), 0, atol=1e-2)


def test_get_ip():
    '''Very simple test to check the ionization potential calculation.'''
    assert get_ip(scf) > 0
    assert_allclose(get_ip(scf), 0.43364808)


@pytest.mark.parametrize('ref, func', [(True, psi),
                                       (False, np.ones_like(psi))])
def test_check_ortho(ref, func):
    '''Test orthogonality check.'''
    assert check_ortho(atoms, func) == ref


@pytest.mark.parametrize('ref, func', [(True, psi),
                                       (False, np.ones_like(psi))])
def test_check_norm(ref, func):
    '''Test normalization check.'''
    assert check_norm(atoms, func) == ref


@pytest.mark.parametrize('ref, func', [(True, psi),
                                       (False, np.ones_like(psi))])
def test_check_orthonorm(ref, func):
    '''Test orthonormalization check.'''
    assert check_orthonorm(atoms, func) == ref


def test_get_isovalue():
    '''Test isovalue calculation.'''
    assert_allclose(get_isovalue(scf.n), 0.025, atol=1e-3)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
