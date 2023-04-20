#!/usr/bin/env python3
'''Test DFT functions.'''
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, SCF
from eminus.dft import get_n_single, get_n_spin, get_n_total, get_psi, H

atoms_unpol = Atoms('Ne', (0, 0, 0), ecut=1, Nspin=1)
atoms_pol = Atoms('Ne', (0, 0, 0), ecut=1, Nspin=2)

scf_unpol = SCF(atoms_unpol)
scf_unpol.run()
scf_pol = SCF(atoms_pol)
scf_pol.run()


@pytest.mark.parametrize('guess', ('gauss', 'rand', 'pseudo'))
@pytest.mark.parametrize('Nspin', (1, 2))
def test_wavefunction(guess, Nspin):
    '''Test the orthonormalization of wave functions.'''
    if Nspin == 1:
        scf = SCF(atoms_unpol, guess=guess)
    else:
        scf = SCF(atoms_pol, guess=guess)
    psi = get_psi(scf, scf.W)
    for s in range(Nspin):
        # Test that WOW is the identity
        ovlp = scf.W[s].conj().T @ scf.atoms.O(scf.W[s])
        assert_allclose(np.diag(ovlp), np.ones(scf.atoms.Nstate))
        # Also test that psiOpsi is the identity
        ovlp = psi[s].conj().T @ scf.atoms.O(psi[s])
        assert_allclose(np.diag(ovlp), np.ones(scf.atoms.Nstate))


@pytest.mark.parametrize('Nspin', (1, 2))
def test_H(Nspin):
    '''Test the Hamiltonian.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    psi = get_psi(scf, scf.W)
    for s in range(Nspin):
        HW = H(scf, s, psi)
        WHW = psi[s].conj().T @ HW
        # Test that WHW is a diagonal matrix
        assert_allclose(WHW, np.diag(np.diag(WHW)), atol=1e-12)


@pytest.mark.parametrize('Nspin', (1, 2))
def test_n_total(Nspin):
    '''Test the total density.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    # Test that the integrated density gives the number of electrons
    n = get_n_total(scf.atoms, scf.Y)
    dV = scf.atoms.Omega / np.prod(scf.atoms.s)
    n_int = np.sum(n) * dV
    assert_allclose(n_int, np.sum(scf.atoms.f))


@pytest.mark.parametrize('Nspin', (1, 2))
def test_n_spin(Nspin):
    '''Test the spin densities.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    # Test that the integrated spin densities gives the number of electrons per spin
    n = get_n_spin(scf.atoms, scf.Y)
    dV = scf.atoms.Omega / np.prod(scf.atoms.s)
    for s in range(Nspin):
        n_int = np.sum(n[s]) * dV
        assert_allclose(n_int, np.sum(scf.atoms.f[s]))


@pytest.mark.parametrize('Nspin', (1, 2))
def test_n_single(Nspin):
    '''Test the single orbital densities.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    # Test that the integrated single orbital densities gives occupation number per orbital
    n = get_n_single(scf.atoms, scf.Y)
    dV = scf.atoms.Omega / np.prod(scf.atoms.s)
    for s in range(Nspin):
        n_int = np.sum(n[s], axis=0) * dV
        assert_allclose(n_int, scf.atoms.f[s])


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
