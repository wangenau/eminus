#!/usr/bin/env python3
"""Test localization functions."""
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, SCF
from eminus.dft import get_psi
from eminus.localizer import get_FLO, get_wannier, wannier_cost
from eminus.tools import check_orthonorm

atoms_unpol = Atoms('CH4', ((0, 0, 0),
                            (1.186, 1.186, 1.186),
                            (1.186, -1.186, -1.186),
                            (-1.186, 1.186, -1.186),
                            (-1.186, -1.186, 1.186)), center=True, ecut=5, Nspin=1)
scf_unpol = SCF(atoms_unpol)
scf_unpol.run()

atoms_pol = Atoms('CH4', ((0, 0, 0),
                          (1.186, 1.186, 1.186),
                          (1.186, -1.186, -1.186),
                          (-1.186, 1.186, -1.186),
                          (-1.186, -1.186, 1.186)), center=True, ecut=5, Nspin=2)
scf_pol = SCF(atoms_pol)
scf_pol.run()

# FODs that will be used for both spin channels
fods = np.array([[9.16, 9.16, 10.89],
                 [10.89, 10.89, 10.89],
                 [10.73, 9.16, 9.16],
                 [9.16, 10.73, 9.16]])


@pytest.mark.parametrize('Nspin', [1, 2])
def test_spread(Nspin):
    """Test the spread calculation."""
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    psi = get_psi(scf, scf.W)
    psi_rs = scf.atoms.I(psi)
    assert check_orthonorm(scf, psi_rs)
    costs = wannier_cost(scf.atoms, psi_rs)
    # The first orbital is a s-type orbital
    assert_allclose(costs[:, 0], 3.6385, atol=5e-4)
    # The others are p-type orbitals with a similar spread
    assert_allclose(costs[:, 1:], 5, atol=0.25)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_flo(Nspin):
    """Test the generation of FLOs."""
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    psi = get_psi(scf, scf.W)
    flo = get_FLO(scf.atoms, psi, [fods] * Nspin)
    assert check_orthonorm(scf, flo)
    costs = wannier_cost(scf.atoms, flo)
    # Check that all transformed orbitals have a similar spread
    # Since the FODs are just a guess and ecut is really small we use a rather large tolerance
    assert_allclose(costs, costs[0, 0], atol=0.05)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_wannier(Nspin):
    """Test the generation of Wannier functions."""
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    psi = get_psi(scf, scf.W)
    # Throw in the flo to prelocalize the orbitals
    flo = get_FLO(scf.atoms, psi, [fods] * Nspin)
    wo = get_wannier(scf.atoms, flo)
    assert check_orthonorm(scf, wo)
    costs = wannier_cost(scf.atoms, wo)
    # Check that all transformed orbitals have a similar spread
    assert_allclose(costs, costs[0, 0], atol=0.0025)


def test_wannier_random_guess():
    """Test the random_guess keyword for Wannier localizations."""
    scf = scf_unpol
    psi = scf.atoms.I(get_psi(scf, scf.W))
    costs = wannier_cost(scf.atoms, psi)
    wo = get_wannier(scf.atoms, psi, Nit=100, random_guess=True, seed=1234)
    assert check_orthonorm(scf, wo)
    assert np.sum(wannier_cost(scf.atoms, wo)) < np.sum(costs)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
