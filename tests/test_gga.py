# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test GGA functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, SCF
from eminus.gga import get_grad_field, get_tau

atoms_unpol = Atoms("He", (0, 0, 0), ecut=1, unrestricted=False)
atoms_unpol.kpts.Nk = 2
atoms_unpol.kpts.k = [[0, 0, 0], [0, 0, 0]]
atoms_unpol.kpts.wk = [0.5, 0.5]
atoms_unpol.kpts.is_built = True
scf_unpol = SCF(atoms_unpol)
scf_unpol.run(betat=1e-4)

atoms_pol = Atoms("He", (0, 0, 0), ecut=1, unrestricted=True)
scf_pol = SCF(atoms_pol)
scf_pol.run()


@pytest.mark.parametrize("unrestricted", [True, False])
def test_get_grad_field(unrestricted):
    """Test the gradient field calculations."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    n_mock = np.zeros_like(scf.n_spin)
    dn_mock = get_grad_field(scf.atoms, n_mock)
    assert isinstance(n_mock, np.ndarray)
    assert dn_mock.shape == (*n_mock.shape, 3)
    assert np.sum(dn_mock) == 0


def test_get_grad_field_type():
    """Test the gradient field return type."""
    scf = scf_unpol
    n_mock = np.zeros_like(scf.n_spin)
    dn_mock = get_grad_field(scf.atoms, n_mock)
    assert dn_mock.dtype == "float64"
    dn_mock = get_grad_field(scf.atoms, n_mock, real=False)
    assert dn_mock.dtype == "complex128"


@pytest.mark.parametrize("unrestricted", [True, False])
def test_get_tau(unrestricted):
    """Test positive-definite kinetic energy density."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    assert scf.Y is not None
    tau = get_tau(scf.atoms, scf.Y)
    T = np.sum(tau) * scf.atoms.dV
    # This integrated KED should be the same as the calculated kinetic energy
    assert_allclose(T, scf.energies.Ekin)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
