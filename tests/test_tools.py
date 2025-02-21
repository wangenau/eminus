# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test tools functions."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

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
    get_dos,
    get_Efermi,
    get_elf,
    get_ip,
    get_isovalue,
    get_magnetization,
    get_multiplicity,
    get_reduced_gradient,
    get_spin_squared,
    get_tautf,
    get_tauw,
    gridspacing2cutoff,
    inertia_tensor,
    orbital_center,
)

atoms = Atoms("He2", ((0, 0, 0), (10, 0, 0)), ecut=5, unrestricted=False, center=True)
scf = SCF(atoms)
scf.run()
psi = atoms.I(get_psi(scf, scf.W))[0]

atoms_unpol = Atoms("He", (0, 0, 0), ecut=1, unrestricted=False)
atoms_pol = Atoms("He", (0, 0, 0), ecut=1, unrestricted=True)
scf_unpol = SCF(atoms_unpol)
scf_unpol.run()
scf_pol = SCF(atoms_pol, guess="sym-rand")
scf_pol.run()

scf_band = copy.deepcopy(scf_unpol)
scf_band.kpts.path = "GX"
scf_band.kpts.Nk = 2
scf_band.converge_bands()


def test_cutoff_and_gridspacing():
    """Test the cutoff and grid spacing conversion."""
    test = np.pi
    out = cutoff2gridspacing(gridspacing2cutoff(test))
    assert_allclose(out, test)


@pytest.mark.parametrize(
    ("coords", "masses", "ref"),
    [(np.eye(3), None, [1 / 3] * 3), (np.eye(3), np.arange(3), [0, 1 / 3, 2 / 3])],
)
def test_center_of_mass(coords, masses, ref):
    """Test the center of mass calculation."""
    out = center_of_mass(coords, masses)
    assert_allclose(out, ref)


@pytest.mark.parametrize("unrestricted", [True, False])
def test_pycom(unrestricted):
    """Test PyCOM routine."""
    atoms = Atoms("He2", ((0, 0, 0), (10, 0, 0)), ecut=2, unrestricted=unrestricted, center=True)
    scf = SCF(atoms)
    scf.run()
    psi = atoms.I(get_psi(scf, scf.W))
    for spin in range(atoms.occ.Nspin):
        assert_allclose(orbital_center(atoms, psi)[0][spin], [[10] * 3] * 2, atol=1e-1)


@pytest.mark.parametrize(
    "coords", [np.array([[1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[0, 0, 1]])]
)
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
    assert get_ip(scf_unpol) > 0
    assert_allclose(get_ip(scf_unpol), 0.39626074)
    assert get_ip(scf_pol) > 0
    assert_allclose(get_ip(scf_pol), 0.39626631)


@pytest.mark.parametrize(("ref", "func"), [(True, psi), (False, np.ones_like(psi))])
def test_check_ortho(ref, func):
    """Test orthogonality check."""
    assert check_ortho(atoms, func) == ref


@pytest.mark.parametrize(("ref", "func"), [(True, psi), (False, np.ones_like(psi))])
def test_check_norm(ref, func):
    """Test normalization check."""
    assert check_norm(atoms, func) == ref


@pytest.mark.parametrize(("ref", "func"), [(True, psi), (False, np.ones_like(psi))])
def test_check_orthonorm(ref, func):
    """Test orthonormalization check."""
    assert check_orthonorm(atoms, func) == ref


def test_get_isovalue():
    """Test isovalue calculation."""
    assert scf.n is not None
    assert_allclose(get_isovalue(scf.n), 0.013, atol=1e-3)


@pytest.mark.parametrize("unrestricted", [True, False])
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


@pytest.mark.parametrize("unrestricted", [True, False])
def test_get_tauw(unrestricted):
    """Test von Weizsaecker kinetic energy density."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
        assert scf.n_spin is not None
        scf.dn_spin = get_grad_field(scf.atoms, scf.n_spin)
    tauw = get_tauw(scf)
    T = np.sum(tauw) * scf.atoms.dV
    # vW KED is exact for one- and two-electron systems
    assert_allclose(T, scf.energies.Ekin, atol=1e-6)


@pytest.mark.parametrize("unrestricted", [True, False])
def test_get_elf(unrestricted):
    """Test electron localization function."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    elf = get_elf(scf)
    # Check the bounds for the ELF
    assert ((elf >= 0) & (elf <= 1)).all()


@pytest.mark.parametrize("unrestricted", [True, False])
def test_get_reduced_gradient(unrestricted):
    """Test reduced density gradient."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
        assert scf.n_spin is not None
        scf.dn_spin = get_grad_field(scf.atoms, scf.n_spin)
    s = get_reduced_gradient(scf, eps=1e-5)
    assert ((s >= 0) & (s < 100)).all()


def test_spin_squared_and_multiplicity():
    """Test the calculation of <S^2> and the multiplicity."""
    atoms = Atoms("H2", ((0, 0, 0), (0, 0, 10)), unrestricted=True, ecut=1)
    rscf = RSCF(atoms)
    assert get_spin_squared(rscf) == 0
    assert get_multiplicity(rscf) == 1

    scf = SCF(atoms, guess="symm-rand")
    scf.run()
    assert_allclose(get_spin_squared(scf), 0, atol=1e-2)
    assert_allclose(get_multiplicity(scf), 1, atol=1e-2)

    scf = SCF(atoms, guess="unsymm-rand")
    scf.run()
    assert_allclose(get_spin_squared(scf), 1, atol=1e-2)
    assert get_multiplicity(scf) > 2


def test_magnetization():
    """Test the calculation of the total magnetization."""
    M = get_magnetization(scf_unpol)
    assert M == 0
    M = get_magnetization(scf_pol)
    assert -1 < M < 1
    atoms = Atoms("H", ((0, 0, 0)), unrestricted=True, ecut=1)
    scf = SCF(atoms)
    scf.run()
    assert_allclose(get_magnetization(scf), 1)


def test_get_Efermi():
    """Test the Fermi energy calculation."""
    Ef = get_Efermi(scf_band)
    assert hasattr(scf_band, "_precomputed")
    e_occ = get_epsilon(scf_band, scf_band.Y, **scf_band._precomputed)
    assert_allclose(Ef, np.max(e_occ))
    scf_band.converge_empty_bands(Nempty=1)
    Ef = get_Efermi(scf_band)
    e_unocc = get_epsilon(scf_band, scf_band.Z, **scf_band._precomputed)
    assert np.max(e_occ) < Ef < np.min(e_unocc)


def test_get_bandgap():
    """Test the band gap calculation."""
    if scf_band.Z is not None:
        scf_band.Z = None
    Eg = get_bandgap(scf_band)
    assert Eg == 0
    scf_band.converge_empty_bands(Nempty=1)
    Eg = get_bandgap(scf_band)
    assert_allclose(Eg, 0.3805155)


def test_get_dos():
    """Test the DOS calculation."""
    assert hasattr(scf_band, "_precomputed")
    e_occ = get_epsilon(scf_band, scf_band.Y, **scf_band._precomputed)
    e, e_dos = get_dos(e_occ, scf_band.kpts.wk, spin=0, npts=500, width=0.1)
    # The maximum of the DOS should be close to the maximum eigenvalue in this case
    assert_allclose(np.max(e_occ), e[np.argmax(e_dos)], atol=1e-2)
    assert np.min(e) < np.min(e_occ)
    assert np.max(e) > np.max(e_occ)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
