# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test DFT functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, Cell, demo, SCF
from eminus.dft import get_n_single, get_n_spin, get_n_total, get_psi, guess_pseudo, guess_random, H
from eminus.tools import get_magnetization

atoms_unpol = Cell("Si", "diamond", 1, 10, kmesh=(2, 1, 1), bands=5)
atoms_pol = Atoms("Ne", (0, 0, 0), ecut=1, unrestricted=True)

scf_unpol = SCF(atoms_unpol)
scf_unpol.run(betat=1e-4)
scf_unpol.converge_bands()
scf_pol = SCF(atoms_pol)
scf_pol.run(betat=1e-4)


@pytest.mark.parametrize("guess", ["random", "pseudo"])
@pytest.mark.parametrize("unrestricted", [True, False])
def test_wavefunction(guess, unrestricted):
    """Test the orthonormalization of wave functions."""
    if unrestricted:
        scf = SCF(atoms_pol, guess=guess)
    else:
        scf = SCF(atoms_unpol, guess=guess)
    if guess == "random":
        W = guess_random(scf)
    elif guess == "pseudo":
        W = guess_pseudo(scf)
    psi = get_psi(scf, W)
    for ik in range(scf.kpts.Nk):
        for s in range(scf.atoms.occ.Nspin):
            # Test that WOW is the identity
            ovlp = W[ik][s].conj().T @ scf.atoms.O(W[ik][s])
            assert_allclose(np.diag(ovlp), np.ones(scf.atoms.occ.Nstate))
            # Also test that psiOpsi is the identity
            ovlp = psi[ik][s].conj().T @ scf.atoms.O(psi[ik][s])
            assert_allclose(np.diag(ovlp), np.ones(scf.atoms.occ.Nstate))


def test_unocc_wavefunction():
    """Test the orthonormalization of unoccupied wave functions."""
    assert scf_unpol.Y is not None
    assert scf_unpol.D is not None
    for ik in range(scf_unpol.kpts.Nk):
        for s in range(scf_unpol.atoms.occ.Nspin):
            # Test that DOD is the identity
            ovlp = scf_unpol.D[ik][s].conj().T @ scf_unpol.atoms.O(scf_unpol.D[ik][s])
            assert_allclose(np.diag(ovlp), np.ones(scf_unpol.atoms.occ.Nempty))
            # Test that the unoccupied wavefunctions are orthogonal to the occupied ones
            assert_allclose(scf_unpol.D[ik][s].conj().T @ scf_unpol.Y[ik][s], 0, atol=1e-15)


@pytest.mark.parametrize("unrestricted", [True, False])
def test_H(unrestricted):
    """Test the Hamiltonian."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    psi = get_psi(scf, scf.W)
    for ik in range(scf.kpts.Nk):
        for s in range(scf.atoms.occ.Nspin):
            HW = H(scf, ik, s, psi)
            WHW = psi[ik][s].conj().T @ HW
            # Test that WHW is a diagonal matrix
            assert_allclose(WHW, np.diag(np.diag(WHW)), atol=1e-12)


@pytest.mark.parametrize("unrestricted", [True, False])
def test_n_total(unrestricted):
    """Test the total density."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    # Test that the integrated density gives the number of electrons
    assert scf.Y is not None
    n = get_n_total(scf.atoms, scf.Y)
    n_int = np.sum(n) * scf.atoms.dV
    assert_allclose(n_int, np.sum(scf.atoms.occ.f * scf.kpts.wk[:, None, None]))


@pytest.mark.parametrize("unrestricted", [True, False])
def test_n_spin(unrestricted):
    """Test the spin densities."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    # Test that the integrated spin densities gives the number of electrons per spin
    assert scf.Y is not None
    n = get_n_spin(scf.atoms, scf.Y)
    for ik in range(scf.kpts.Nk):
        for s in range(scf.atoms.occ.Nspin):
            n_int = np.sum(n[s]) * scf.atoms.dV
            assert_allclose(n_int, np.sum(scf.atoms.occ.f[ik, s]))


@pytest.mark.parametrize("unrestricted", [True, False])
def test_n_single(unrestricted):
    """Test the single orbital densities."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    # Test that the integrated single orbital densities gives occupation number per orbital
    assert scf.Y is not None
    n = get_n_single(scf.atoms, scf.Y)
    for ik in range(scf.kpts.Nk):
        for s in range(scf.atoms.occ.Nspin):
            n_int = np.sum(n[s], axis=0) * scf.atoms.dV
            assert_allclose(n_int, scf.atoms.occ.f[ik, s])


def test_k_point_permutation():
    """Check that the order of k-points does not change the calculation."""
    cell = Cell("C", "diamond", 10, 6.75, kmesh=(2, 1, 1))
    scf = SCF(cell)
    etot1 = scf.run()
    assert hasattr(cell.kpts, "_k")
    cell.kpts._k = cell.kpts._k[::-1]
    assert not np.all(cell.kpts.k == scf.kpts.k)
    scf = SCF(cell)
    etot2 = scf.run()
    assert_allclose(etot1, etot2, atol=1e-7)


def test_demo():
    """Test that the demo function works without problems."""
    demo()


def test_magnetization():
    """Check that the order of k-points does not change the calculation."""
    magnetization = 0.5
    cell = Cell("Al", "fcc", 1, 8, unrestricted=True, magnetization=magnetization)
    scf = SCF(cell)
    scf.run()
    assert_allclose(get_magnetization(scf), cell.occ.magnetization)
    cell = Cell("Al", "fcc", 1, 8, unrestricted=True, bands=4, smearing=1, magnetization=1)
    scf = SCF(cell)
    scf.run()
    assert scf.atoms.occ.magnetization < 1
    assert not np.array_equal(cell.occ.f, scf.atoms.occ.f)
    assert_allclose(get_magnetization(scf), scf.atoms.occ.magnetization, atol=1e-7)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
