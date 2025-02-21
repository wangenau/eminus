# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test different energy contributions."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, Cell, SCF
from eminus.dft import get_epsilon, guess_pseudo
from eminus.energies import Energy, get_Eband, get_Eentropy
from eminus.minimizer import scf_step

# The reference contributions are similar for the polarized and unpolarized case,
# but not necessary the same (for bad numerics)
E_ref = {
    "Ekin": 12.034547463,
    "Ecoul": 17.843671711,
    "Exc": -4.242216786,
    "Eloc": -58.537381335,
    "Enonloc": 8.152508709,
    "Eewald": -4.539675967,
    "Esic": -0.397094829,
    "Etot": -29.685641033,
}

# Run the spin-unpolarized calculation at first
atoms_unpol = Atoms("Ne", (0, 0, 0), ecut=10, unrestricted=False)
atoms_unpol.s = 20
scf_unpol = SCF(atoms_unpol, sic=True)
scf_unpol.run()
# Do the spin-polarized calculation afterwards
# Use the orbitals from the restricted calculation as an initial guess for the unrestricted case
# This saves time and ensures we run into the same minimum
atoms_pol = Atoms("Ne", (0, 0, 0), ecut=10, unrestricted=True)
atoms_pol.s = 20
scf_pol = SCF(atoms_pol, sic=True)
assert scf_unpol.W is not None
scf_pol.W = [np.array([scf_unpol.W[0][0] / 2, scf_unpol.W[0][0] / 2])]
scf_pol.run()


@pytest.mark.parametrize("energy", E_ref.keys())
def test_energies_unpol(energy):
    """Check the spin-unpolarized energy contributions."""
    E = getattr(scf_unpol.energies, energy)
    assert_allclose(E, E_ref[energy], atol=1e-4)


@pytest.mark.parametrize("energy", E_ref.keys())
def test_energies_pol(energy):
    """Check the spin-polarized energy contributions."""
    E = getattr(scf_pol.energies, energy)
    assert_allclose(E, E_ref[energy], atol=1e-4)


def test_mgga_sic_unpol():
    """Check the spin-unpolarized SIC energy for meta-GGAs."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    scf_tmp = copy.deepcopy(scf_unpol)
    scf_tmp.xc = ":mgga_x_scan,:mgga_c_scan"
    scf_tmp.opt = {"auto": 1}
    scf_tmp.run()
    assert_allclose(scf_tmp.energies.Esic, -0.6092, atol=1e-4)


def test_mgga_sic_pol():
    """Check the spin-polarized SIC energy for meta-GGAs."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    scf_tmp = copy.deepcopy(scf_pol)
    scf_tmp.xc = ":mgga_x_scan,:mgga_c_scan"
    scf_tmp.opt = {"auto": 1}
    scf_tmp.run()
    assert_allclose(scf_tmp.energies.Esic, -0.5515, atol=1e-4)


def test_get_Eband_unpol():
    """Check the spin-unpolarized band energy."""
    assert scf_unpol.Y is not None
    Eband = get_Eband(scf_unpol, scf_unpol.Y)
    assert_allclose(Eband, -4.1123, atol=1e-4)


def test_get_Eband_pol():
    """Check the spin-polarized band energy."""
    assert scf_pol.Y is not None
    assert hasattr(scf_pol, "_precomputed")
    Eband = get_Eband(scf_pol, scf_pol.Y, **scf_pol._precomputed)
    # About twice as large as the unpolarized case since we do not account for occupations
    # The "real" energy does not matter, we only want to minimize the band energy
    assert_allclose(Eband, -8.2246, atol=1e-4)


def test_get_Eentropy_unpol():
    """Check the spin-unpolarized entropic energy."""
    scf_unpol.atoms.occ.smearing = 1
    epsilon = get_epsilon(scf_unpol, scf_unpol.W)
    Eband = get_Eentropy(scf_unpol, epsilon, 0)
    assert_allclose(Eband, -4.4395, atol=1e-4)


def test_get_Eentropy_pol():
    """Check the spin-polarized entropic energy."""
    scf_pol.atoms.occ.smearing = 1
    epsilon = get_epsilon(scf_pol, scf_pol.W)
    Eband = get_Eentropy(scf_pol, epsilon, 0)
    assert_allclose(Eband, -4.4395, atol=1e-4)


def test_multiple_k():
    """Test that the energy for one or multiple, identical k-points is the same."""
    atoms = Cell("Si", "diamond", ecut=30, a=10.2631)
    scf = SCF(atoms, etol=1e-5)
    scf.W = guess_pseudo(scf)
    scf.dn_spin, scf.tau = None, None
    scf_step(scf, 0)

    atoms = Cell("Si", "diamond", ecut=30, a=10.2631, kmesh=(2, 2, 2))
    atoms.set_k(np.zeros((8, 3)))
    scf_k = SCF(atoms, etol=1e-5)
    scf_k.W = guess_pseudo(scf_k)
    scf_k.dn_spin, scf_k.tau = None, None
    scf_step(scf_k, 0)

    assert_allclose(scf.energies.Etot, scf_k.energies.Etot)


def test_extrapolate():
    """Test the extrapolation function."""
    e = Energy()
    e.Ekin = 2
    assert e.extrapolate() == 2
    e.Eentropy = 2
    assert e.extrapolate() == 3


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
