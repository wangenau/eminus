#!/usr/bin/env python3
"""Test DFT functions."""
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, demo, SCF
from eminus.dft import get_n_single, get_n_spin, get_n_total, get_psi, guess_pseudo, guess_random, H

atoms_unpol = Atoms('Ne', (0, 0, 0), ecut=1, unrestricted=False)
atoms_pol = Atoms('Ne', (0, 0, 0), ecut=1, unrestricted=True)

scf_unpol = SCF(atoms_unpol)
scf_unpol.run()
scf_pol = SCF(atoms_pol)
scf_pol.run()


@pytest.mark.parametrize('guess', ['random', 'pseudo'])
@pytest.mark.parametrize('unrestricted', [True, False])
def test_wavefunction(guess, unrestricted):
    """Test the orthonormalization of wave functions."""
    if unrestricted:
        scf = SCF(atoms_pol, guess=guess)
    else:
        scf = SCF(atoms_unpol, guess=guess)
    if guess == 'random':
        W = guess_random(scf)
    elif guess == 'pseudo':
        W = guess_pseudo(scf)
    psi = get_psi(scf, W)
    for s in range(scf.atoms.occ.Nspin):
        # Test that WOW is the identity
        ovlp = W[s].conj().T @ scf.atoms.O(W[s])
        assert_allclose(np.diag(ovlp), np.ones(scf.atoms.occ.Nstate))
        # Also test that psiOpsi is the identity
        ovlp = psi[s].conj().T @ scf.atoms.O(psi[s])
        assert_allclose(np.diag(ovlp), np.ones(scf.atoms.occ.Nstate))


@pytest.mark.parametrize('unrestricted', [True, False])
def test_H(unrestricted):
    """Test the Hamiltonian."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    psi = get_psi(scf, scf.W)
    for s in range(scf.atoms.occ.Nspin):
        HW = H(scf, s, psi)
        WHW = psi[s].conj().T @ HW
        # Test that WHW is a diagonal matrix
        assert_allclose(WHW, np.diag(np.diag(WHW)), atol=1e-12)


@pytest.mark.parametrize('unrestricted', [True, False])
def test_n_total(unrestricted):
    """Test the total density."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    # Test that the integrated density gives the number of electrons
    n = get_n_total(scf.atoms, scf.Y)
    n_int = np.sum(n) * scf.atoms.dV
    assert_allclose(n_int, np.sum(scf.atoms.occ.f))


@pytest.mark.parametrize('unrestricted', [True, False])
def test_n_spin(unrestricted):
    """Test the spin densities."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    # Test that the integrated spin densities gives the number of electrons per spin
    n = get_n_spin(scf.atoms, scf.Y)
    for s in range(scf.atoms.occ.Nspin):
        n_int = np.sum(n[s]) * scf.atoms.dV
        assert_allclose(n_int, np.sum(scf.atoms.occ.f[s]))


@pytest.mark.parametrize('unrestricted', [True, False])
def test_n_single(unrestricted):
    """Test the single orbital densities."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    # Test that the integrated single orbital densities gives occupation number per orbital
    n = get_n_single(scf.atoms, scf.Y)
    for s in range(scf.atoms.occ.Nspin):
        n_int = np.sum(n[s], axis=0) * scf.atoms.dV
        assert_allclose(n_int, scf.atoms.occ.f[s])


def test_demo():
    """Test that the demo function works without problems."""
    demo()


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
