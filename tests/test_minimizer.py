# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test different minimization functions."""

import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, SCF
from eminus.minimizer import IMPLEMENTED

E_ref = -1.899358114  # The reference value is the same for the polarized and unpolarized case
tolerance = 1e-9
atoms_unpol = Atoms("He", (0, 0, 0), ecut=2, unrestricted=False, verbose="debug")
atoms_pol = Atoms("He", (0, 0, 0), ecut=2, unrestricted=True, verbose="debug")
# Restart the calculations from a previous result
# This way we make sure we run into the same minima and we have no convergence problems
scf_unpol = SCF(atoms_unpol, etol=tolerance)
scf_pol = SCF(atoms_pol, etol=tolerance)


@pytest.mark.parametrize("minimizer", IMPLEMENTED.keys())
def test_minimizer_unpol(minimizer):
    """Check the spin-unpolarized minimizer functions."""
    scf_unpol.opt = {minimizer: 110}
    if minimizer in {"cg", "pccg", "auto"}:
        for i in range(1, 5):
            E = scf_unpol.run(cgform=i)
            assert_allclose(E, E_ref, atol=tolerance)
    else:
        E = scf_unpol.run()
        assert_allclose(E, E_ref, atol=tolerance)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # Filter an occurring overflow warning
@pytest.mark.parametrize("minimizer", IMPLEMENTED.keys())
def test_minimizer_pol(minimizer):
    """Check the spin-polarized minimizer functions."""
    scf_pol.opt = {minimizer: 110}
    if minimizer in {"cg", "pccg", "auto"}:
        for i in range(1, 5):
            E = scf_pol.run(cgform=i)
            assert_allclose(E, E_ref, atol=tolerance)
    else:
        E = scf_pol.run()
        assert_allclose(E, E_ref, atol=tolerance)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
