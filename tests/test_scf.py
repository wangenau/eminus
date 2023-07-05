#!/usr/bin/env python3
"""Test the SCF class."""
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, SCF
from eminus.tools import center_of_mass

atoms = Atoms('He', (0, 0, 0), s=10, Nspin=2)


def test_atoms():
    """Test that the Atoms object is independent."""
    scf = SCF(atoms)
    assert id(scf.atoms) != id(atoms)


def test_xc():
    """Test that xc functionals are correctly parsed."""
    scf = SCF(atoms, xc='LDA,VWN')
    assert scf.xc == ['lda_x', 'lda_c_vwn']
    scf = SCF(atoms, xc=',')
    assert scf.xc == ['mock_xc', 'mock_xc']


def test_pot():
    """Test that potentials are correctly parsed and initialized."""
    scf = SCF(atoms, pot='GTH')
    assert not [x for x in (scf.Vloc, scf.NbetaNL, scf.prj2beta, scf.betaNL) if x is None]
    scf = SCF(atoms, pot='GE')
    assert scf.Vloc is not None
    assert [x for x in (scf.NbetaNL, scf.prj2beta, scf.betaNL) if x is None]


def test_guess():
    """Test initialization of the guess method."""
    scf = SCF(atoms, guess='RAND')
    assert scf.guess == 'rand'
    scf = SCF(atoms, guess='bogus')


def test_gradtol():
    """Test the convergence depending of the gradient norm."""
    scf = SCF(atoms, etol=1, gradtol=1e-2)
    scf.run()
    assert scf.energies.Etot < -1


def test_sic():
    """Test that the SIC routine runs."""
    scf = SCF(atoms, xc='pbe', min={'sd': 1}, sic=True)
    scf.run()
    assert scf.energies.Esic != 0


def test_symmetric():
    """Test the symmetry option for H2 dissociation."""
    atoms = Atoms('H2', ((0, 0, 0), (0, 0, 6)), Nspin=2, ecut=1)
    e_symm = SCF(atoms, symmetric=True).run()
    e_unsymm = SCF(atoms, symmetric=False).run()
    assert e_symm > e_unsymm


def test_verbose():
    """Test the verbosity level."""
    scf = SCF(atoms)
    assert scf.verbose == atoms.verbose
    assert scf.log.verbose == atoms.log.verbose
    level = 'DEBUG'
    scf = SCF(atoms, verbose=level)
    assert scf.verbose == level
    assert scf.log.verbose == level


def test_clear():
    """Test the clear function."""
    scf = SCF(atoms, min={'sd': 1})
    scf.run()
    scf.clear()
    assert not scf.is_converged
    assert [x for x in (scf.Y, scf.n_spin, scf.dn_spin, scf.phi, scf.exc, scf.vxc) if x is None]


@pytest.mark.parametrize('center', [None, atoms.a / 2])
def test_recenter(center):
    """Test the recenter function."""
    scf = SCF(atoms)
    scf.run()
    assert scf.is_converged
    scf.recenter(center)
    W = atoms.I(scf.W)
    com = center_of_mass(scf.atoms.X)
    # Check that the density is centered around the atom
    assert_allclose(center_of_mass(atoms.r, scf.n), com, atol=1e-3)
    # Check that the orbitals are centered around the atom
    assert_allclose(center_of_mass(atoms.r, W[0, :, 0].conj() * W[0, :, 0]), com, atol=0.00125)
    assert_allclose(center_of_mass(atoms.r, W[1, :, 0].conj() * W[1, :, 0]), com, atol=0.00125)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
