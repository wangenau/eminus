# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test localization functions."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, SCF
from eminus.dft import get_psi
from eminus.localizer import get_FLO, get_scdm, get_wannier, wannier_cost
from eminus.tools import check_orthonorm

atoms_unpol = Atoms(
    "CH4",
    (
        (0, 0, 0),
        (1.186, 1.186, 1.186),
        (1.186, -1.186, -1.186),
        (-1.186, 1.186, -1.186),
        (-1.186, -1.186, 1.186),
    ),
    center=True,
    ecut=5,
    unrestricted=False,
)
scf_unpol = SCF(atoms_unpol)
scf_unpol.run()

atoms_pol = Atoms(
    "CH4",
    (
        (0, 0, 0),
        (1.186, 1.186, 1.186),
        (1.186, -1.186, -1.186),
        (-1.186, 1.186, -1.186),
        (-1.186, -1.186, 1.186),
    ),
    center=True,
    ecut=5,
    unrestricted=True,
)
scf_pol = SCF(atoms_pol)
scf_pol.run()

# FODs that will be used for both spin channels
fods = np.array(
    [
        [9.16, 9.16, 10.89],
        [10.89, 10.89, 10.89],
        [10.73, 9.16, 9.16],
        [9.16, 10.73, 9.16],
    ]
)


@pytest.mark.parametrize("unrestricted", [True, False])
def test_spread(unrestricted):
    """Test the spread calculation."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    psi = get_psi(scf, scf.W)
    psi_rs = scf.atoms.I(psi)[0]
    assert check_orthonorm(scf, psi_rs)
    costs = wannier_cost(scf.atoms, psi_rs)
    # The first orbital is a s-type orbital
    assert_allclose(costs[:, 0], 3.6385, atol=5e-4)
    # The others are p-type orbitals with a similar spread
    assert_allclose(costs[:, 1:], 5, atol=0.25)


@pytest.mark.parametrize("unrestricted", [True, False])
def test_flo(unrestricted):
    """Test the generation of FLOs."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    psi = get_psi(scf, scf.W)
    flo = get_FLO(scf.atoms, psi, [fods] * scf.atoms.occ.Nspin)[0]
    assert check_orthonorm(scf, flo)
    costs = wannier_cost(scf.atoms, flo)
    # Check that all transformed orbitals have a similar spread
    # Since the FODs are just a guess and ecut is really small we use a rather large tolerance
    assert_allclose(costs, costs[0, 0], atol=0.05)


@pytest.mark.parametrize("unrestricted", [True, False])
def test_wannier(unrestricted):
    """Test the generation of Wannier functions."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    psi = get_psi(scf, scf.W)
    # Throw in the SCDMs to prelocalize the orbitals
    scdm = get_scdm(scf.atoms, psi)
    wo = get_wannier(scf.atoms, scdm)[0]
    assert check_orthonorm(scf, wo)
    costs = wannier_cost(scf.atoms, wo)
    # Check that all transformed orbitals have a similar spread
    assert_allclose(costs, costs[0, 0], atol=0.001)
    scf_tmp = copy.deepcopy(scf)
    scf_tmp.atoms.a = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    assert_allclose(get_wannier(scf_tmp.atoms, scdm), scdm)


def test_wannier_random_guess():
    """Test the random_guess keyword for Wannier localizations."""
    scf = scf_unpol
    psi = scf.atoms.I(get_psi(scf, scf.W))[0]
    costs = wannier_cost(scf.atoms, psi)
    wo = get_wannier(scf.atoms, psi, Nit=100, random_guess=True, seed=1234)
    assert check_orthonorm(scf, wo)
    assert np.sum(wannier_cost(scf.atoms, wo)) < np.sum(costs)


@pytest.mark.parametrize("unrestricted", [True, False])
def test_scdm(unrestricted):
    """Test the generation of SCDM localized orbitals."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    psi = get_psi(scf, scf.W)
    scdm = get_scdm(scf.atoms, psi)[0]
    assert check_orthonorm(scf, scdm)
    costs = wannier_cost(scf.atoms, scdm)
    # Check that all transformed orbitals roughly a similar spread
    assert_allclose(costs, costs[0, 0], atol=0.2)
    # Check that the SCDM orbitals have a lower spread than the KS orbitals
    assert np.sum(costs) < np.sum(wannier_cost(scf.atoms, scf.atoms.I(psi)))


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
