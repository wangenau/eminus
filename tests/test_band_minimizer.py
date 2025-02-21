# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test different band minimization functions."""

import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, SCF
from eminus.energies import get_Eband
from eminus.minimizer import IMPLEMENTED

# The reference the polarized and unpolarized case is not the same since the band energies are not a
# physical property, only something we want to minimize
E_ref_unpol = -0.438666603
E_ref_pol = -0.877315883
tolerance = 1e-9
atoms_unpol = Atoms("He", (0, 0, 0), ecut=2, unrestricted=False, verbose="debug")
atoms_pol = Atoms("He", (0, 0, 0), ecut=2, unrestricted=True, verbose="debug")
# Restart the calculations from a previous result
# This way we make sure we run into the same minima and we have no convergence problems
scf_unpol = SCF(atoms_unpol, etol=tolerance)
scf_unpol.run()
scf_pol = SCF(atoms_pol, etol=tolerance)
scf_pol.run()


@pytest.mark.parametrize("minimizer", IMPLEMENTED.keys())
def test_minimizer_unpol(minimizer):
    """Check the spin-unpolarized band minimizer functions."""
    scf_unpol.opt = {minimizer: 100}
    assert scf_unpol.Y is not None
    assert hasattr(scf_unpol, "_precomputed")
    if minimizer in {"cg", "pccg", "auto"}:
        for i in range(1, 5):
            scf_unpol.converge_bands(cgform=i)
            E = get_Eband(scf_unpol, scf_unpol.Y, **scf_unpol._precomputed)
            assert_allclose(E, E_ref_unpol, atol=tolerance)
    else:
        scf_unpol.converge_bands()
        E = get_Eband(scf_unpol, scf_unpol.Y, **scf_unpol._precomputed)
        assert_allclose(E, E_ref_unpol, atol=tolerance)


@pytest.mark.parametrize("minimizer", IMPLEMENTED.keys())
def test_minimizer_pol(minimizer):
    """Check the spin-polarized band minimizer functions."""
    scf_pol.opt = {minimizer: 100}
    assert scf_pol.Y is not None
    assert hasattr(scf_pol, "_precomputed")
    if minimizer in {"cg", "pccg", "auto"}:
        for i in range(1, 5):
            scf_pol.converge_bands(cgform=i)
            E = get_Eband(scf_pol, scf_pol.Y, **scf_pol._precomputed)
            assert_allclose(E, E_ref_pol, atol=tolerance)
    else:
        scf_pol.converge_bands()
        E = get_Eband(scf_pol, scf_pol.Y, **scf_pol._precomputed)
        assert_allclose(E, E_ref_pol, atol=tolerance)


def test_empty_W():
    """Test the band minimization for a few edge cases."""
    scf_unpol.is_converged = False
    scf_unpol.W = None
    scf_unpol.guess = "pseudo"
    scf_unpol.opt = {"auto": 1}
    scf_unpol.converge_bands()
    scf_unpol.converge_empty_bands(1)
    scf_unpol.opt = {"pccg": 2}
    scf_unpol.converge_bands()


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
