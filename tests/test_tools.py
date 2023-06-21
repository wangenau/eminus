#!/usr/bin/env python3
'''Test tools functions.'''
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, RSCF, SCF
from eminus.dft import get_psi
from eminus.tools import (center_of_mass, check_norm, check_ortho, check_orthonorm,
                          cutoff2gridspacing, get_dipole, get_elf, get_ip, get_isovalue,
                          get_multiplicity, get_reduced_gradient, get_spin_squared, get_tautf,
                          get_tauw, gridspacing2cutoff, inertia_tensor, orbital_center)


min = {'sd': 25, 'pccg': 25}
atoms = Atoms('He2', ((0, 0, 0), (10, 0, 0)), s=15, Nspin=1, center=True)
scf = SCF(atoms, min=min)
scf.run()
psi = atoms.I(get_psi(scf, scf.W))

atoms_unpol = Atoms('He', (0, 0, 0), ecut=1, Nspin=1)
atoms_pol = Atoms('He', (0, 0, 0), ecut=1, Nspin=2)
scf_unpol = SCF(atoms_unpol, min=min)
scf_unpol.run()
scf_pol = SCF(atoms_pol, min=min)
scf_pol.run()


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


@pytest.mark.parametrize('Nspin', (1, 2))
def test_pycom(Nspin):
    '''Test PyCOM routine.'''
    atoms = Atoms('He2', ((0, 0, 0), (10, 0, 0)), s=10, Nspin=Nspin, center=True).build()
    scf = SCF(atoms, min=min)
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
    assert_allclose(get_ip(scf), 0.43364823)


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


@pytest.mark.parametrize('Nspin', (1, 2))
def test_get_tautf(Nspin):
    '''Test Thomas-Fermi kinetic energy density.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    tautf = get_tautf(scf)
    T = np.sum(tautf) * scf.atoms.Omega / np.prod(scf.atoms.s)
    # TF KED is not exact, but similar for simple systems
    assert_allclose(T, scf.energies.Ekin, atol=0.2)


@pytest.mark.parametrize('Nspin', (1, 2))
def test_get_tauw(Nspin):
    '''Test von Weizsaecker kinetic energy density.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    tauw = get_tauw(scf)
    T = np.sum(tauw) * scf.atoms.Omega / np.prod(scf.atoms.s)
    # vW KED is exact for one- and two-electron systems
    assert_allclose(T, scf.energies.Ekin, atol=1e-6)


@pytest.mark.parametrize('Nspin', (1, 2))
def test_get_elf(Nspin):
    '''Test electron localization function.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    elf = get_elf(scf)
    # Check the bounds for the ELF
    assert ((0 <= elf) & (elf <= 1)).all()


@pytest.mark.parametrize('Nspin', (1, 2))
def test_get_reduced_gradient(Nspin):
    '''Test reduced density gradient.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    s = get_reduced_gradient(scf, eps=1e-5)
    assert ((0 <= s) & (s < 100)).all()


def test_spin_squared_and_multiplicity():
    '''Test the calculation of <S^2> and the multiplicity.'''
    atoms = Atoms('H2', ((0, 0, 0), (0, 0, 10)), Nspin=2, ecut=1)
    rscf = RSCF(atoms)
    assert get_spin_squared(rscf) == 0
    assert get_multiplicity(rscf) == 1

    scf = SCF(atoms, symmetric=True)
    scf.run()
    assert get_spin_squared(scf) == 0
    assert get_multiplicity(scf) == 1

    scf = SCF(atoms, symmetric=False)
    scf.run()
    assert_allclose(get_spin_squared(scf), 1, atol=1e-2)
    assert get_multiplicity(scf) > 2


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
