#!/usr/bin/env python3
"""Test tools functions."""
import copy

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, RSCF, SCF
from eminus.dft import get_epsilon, get_psi
from eminus.gga import get_grad_field
from eminus.tools import (
    center_of_mass,
    check_norm,
    check_ortho,
    check_orthonorm,
    cutoff2gridspacing,
    get_bandgap,
    get_dipole,
    get_Efermi,
    get_elf,
    get_ip,
    get_isovalue,
    get_multiplicity,
    get_reduced_gradient,
    get_spin_squared,
    get_tautf,
    get_tauw,
    gridspacing2cutoff,
    inertia_tensor,
    orbital_center,
)

opt = {'sd': 25, 'pccg': 25}
atoms = Atoms('He2', ((0, 0, 0), (10, 0, 0)), ecut=5, unrestricted=False, center=True)
scf = SCF(atoms, opt=opt)
scf.run()
psi = atoms.I(get_psi(scf, scf.W))[0]

atoms_unpol = Atoms('He', (0, 0, 0), ecut=1, unrestricted=False)
atoms_pol = Atoms('He', (0, 0, 0), ecut=1, unrestricted=True)
scf_unpol = SCF(atoms_unpol, opt=opt)
scf_unpol.run()
scf_pol = SCF(atoms_pol, opt=opt)
scf_pol.run()

scf_band = copy.deepcopy(scf_unpol)
scf_band.kpts.path = 'GX'
scf_band.kpts.Nk = 2
scf_band.converge_bands()


def test_cutoff_and_gridspacing():
    """Test the cutoff and grid spacing conversion."""
    rng = default_rng()
    test = np.abs(rng.random(100))
    out = cutoff2gridspacing(gridspacing2cutoff(test))
    assert_allclose(out, test)


@pytest.mark.parametrize(('coords', 'masses', 'ref'), [
    (np.eye(3), None, [1 / 3] * 3),
    (np.eye(3), np.arange(3), [0, 1 / 3, 2 / 3])])
def test_center_of_mass(coords, masses, ref):
    """Test the center of mass calculation."""
    out = center_of_mass(coords, masses)
    assert_allclose(out, ref)


@pytest.mark.parametrize('unrestricted', [True, False])
def test_pycom(unrestricted):
    """Test PyCOM routine."""
    atoms = Atoms('He2', ((0, 0, 0), (10, 0, 0)), ecut=1, unrestricted=unrestricted, center=True)
    scf = SCF(atoms, opt=opt)
    scf.run()
    psi = atoms.I(get_psi(scf, scf.W))
    for spin in range(atoms.occ.Nspin):
        assert_allclose(orbital_center(atoms, psi)[0][spin], [[10] * 3] * 2, atol=1e-1)


@pytest.mark.parametrize('coords', [np.array([[1, 0, 0]]),
                                    np.array([[0, 1, 0]]),
                                    np.array([[0, 0, 1]])])
def test_inertia_tensor(coords):
    """Test the inertia tensor calculation."""
    out = inertia_tensor(coords)
    ref = np.eye(3)
    ref[np.nonzero(coords[0])] = [0] * 3
    assert_allclose(out, ref)


def test_get_dipole():
    """Test the electric dipole moment calculation."""
    assert_allclose(get_dipole(scf), 0, atol=1e-2)


def test_get_ip():
    """Very simple test to check the ionization potential calculation."""
    assert get_ip(scf) > 0
    assert_allclose(get_ip(scf), 0.4873204)


@pytest.mark.parametrize(('ref', 'func'), [(True, psi),
                                           (False, np.ones_like(psi))])
def test_check_ortho(ref, func):
    """Test orthogonality check."""
    assert check_ortho(atoms, func) == ref


@pytest.mark.parametrize(('ref', 'func'), [(True, psi),
                                           (False, np.ones_like(psi))])
def test_check_norm(ref, func):
    """Test normalization check."""
    assert check_norm(atoms, func) == ref


@pytest.mark.parametrize(('ref', 'func'), [(True, psi),
                                           (False, np.ones_like(psi))])
def test_check_orthonorm(ref, func):
    """Test orthonormalization check."""
    assert check_orthonorm(atoms, func) == ref


def test_get_isovalue():
    """Test isovalue calculation."""
    assert_allclose(get_isovalue(scf.n), 0.013, atol=1e-3)


@pytest.mark.parametrize('unrestricted', [True, False])
def test_get_tautf(unrestricted):
    """Test Thomas-Fermi kinetic energy density."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    tautf = get_tautf(scf)
    T = np.sum(tautf) * scf.atoms.dV
    # TF KED is not exact, but similar for simple systems
    assert_allclose(T, scf.energies.Ekin, atol=0.2)


@pytest.mark.parametrize('unrestricted', [True, False])
def test_get_tauw(unrestricted):
    """Test von Weizsaecker kinetic energy density."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
        scf.dn_spin = get_grad_field(scf.atoms, scf.n_spin)
    tauw = get_tauw(scf)
    T = np.sum(tauw) * scf.atoms.dV
    # vW KED is exact for one- and two-electron systems
    assert_allclose(T, scf.energies.Ekin, atol=1e-6)


@pytest.mark.parametrize('unrestricted', [True, False])
def test_get_elf(unrestricted):
    """Test electron localization function."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    elf = get_elf(scf)
    # Check the bounds for the ELF
    assert ((elf >= 0) & (elf <= 1)).all()


@pytest.mark.parametrize('unrestricted', [True, False])
def test_get_reduced_gradient(unrestricted):
    """Test reduced density gradient."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
        scf.dn_spin = get_grad_field(scf.atoms, scf.n_spin)
    s = get_reduced_gradient(scf, eps=1e-5)
    assert ((s >= 0) & (s < 100)).all()


def test_spin_squared_and_multiplicity():
    """Test the calculation of <S^2> and the multiplicity."""
    atoms = Atoms('H2', ((0, 0, 0), (0, 0, 10)), unrestricted=True, ecut=1)
    rscf = RSCF(atoms)
    assert get_spin_squared(rscf) == 0
    assert get_multiplicity(rscf) == 1

    scf = SCF(atoms, guess='symm-rand')
    scf.run()
    assert get_spin_squared(scf) == 0
    assert get_multiplicity(scf) == 1

    scf = SCF(atoms, guess='unsymm-rand')
    scf.run()
    assert_allclose(get_spin_squared(scf), 1, atol=1e-2)
    assert get_multiplicity(scf) > 2


def test_get_Efermi():
    """Test the Fermi energy calculation."""
    Ef = get_Efermi(scf_band)
    e_occ = get_epsilon(scf_band, scf_band.Y, **scf_band._precomputed)
    assert_allclose(Ef, np.max(e_occ))
    scf_band.converge_empty_bands(Nempty=1)
    Ef = get_Efermi(scf_band)
    e_unocc = get_epsilon(scf_band, scf_band.Z, **scf_band._precomputed)
    assert np.max(e_occ) < Ef < np.min(e_unocc)


def test_get_bandgap():
    """Test the band gap calculation."""
    if hasattr(scf_band, 'Z'):
        del scf_band.Z
    Eg = get_bandgap(scf_band)
    assert Eg == 0
    scf_band.converge_empty_bands(Nempty=1)
    Eg = get_bandgap(scf_band)
    assert_allclose(Eg, 0.380516)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
