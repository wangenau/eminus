#!/usr/bin/env python3
"""Test GGA functions."""
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, SCF
from eminus.gga import get_grad_field, get_tau

atoms_unpol = Atoms('He', (0, 0, 0), ecut=1, unrestricted=False)
atoms_pol = Atoms('He', (0, 0, 0), ecut=1, unrestricted=True)

scf_unpol = SCF(atoms_unpol)
scf_unpol.run()
scf_pol = SCF(atoms_pol)
scf_pol.run()


@pytest.mark.parametrize('unrestricted', [True, False])
def test_get_grad_field(unrestricted):
    """Test the gradient field calculations."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    n_mock = np.zeros_like(scf.n_spin)
    dn_mock = get_grad_field(scf.atoms, n_mock)
    assert dn_mock.shape == n_mock.shape + (3,)
    assert np.sum(dn_mock) == 0


def test_get_grad_field_type():
    """Test the gradient field return type."""
    scf = scf_unpol
    n_mock = np.zeros_like(scf.n_spin)
    dn_mock = get_grad_field(scf.atoms, n_mock)
    assert dn_mock.dtype == 'float64'
    dn_mock = get_grad_field(scf.atoms, n_mock, real=False)
    assert dn_mock.dtype == 'complex128'


@pytest.mark.parametrize('unrestricted', [True, False])
def test_get_tau(unrestricted):
    """Test positive-definite kinetic energy density."""
    if unrestricted:
        scf = scf_pol
    else:
        scf = scf_unpol
    tau = get_tau(scf.atoms, scf.Y)
    T = np.sum(tau) * scf.atoms.dV
    # This integrated KED should be the same as the calculated kinetic energy
    assert_allclose(T, scf.energies.Ekin)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
