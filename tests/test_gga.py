#!/usr/bin/env python3
'''Test GGA functions.'''
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, SCF
from eminus.gga import get_grad_field, get_tau

atoms_unpol = Atoms('He', (0, 0, 0), ecut=1, Nspin=1)
atoms_pol = Atoms('He', (0, 0, 0), ecut=1, Nspin=2)

scf_unpol = SCF(atoms_unpol)
scf_unpol.run()
scf_pol = SCF(atoms_pol)
scf_pol.run()


@pytest.mark.parametrize('Nspin', [1, 2])
def test_get_grad_field(Nspin):
    '''Test the gradient field calculations.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    n_mock = np.zeros_like(scf.n_spin)
    dn_mock = get_grad_field(scf.atoms, n_mock)
    assert dn_mock.shape == n_mock.shape + (3,)
    assert np.sum(dn_mock) == 0


@pytest.mark.parametrize('Nspin', [1, 2])
def test_get_tau(Nspin):
    '''Test positive-definite kinetic energy density.'''
    if Nspin == 1:
        scf = scf_unpol
    else:
        scf = scf_pol
    tau = get_tau(scf.atoms, scf.Y)
    T = np.sum(tau) * scf.atoms.Omega / np.prod(scf.atoms.s)
    # This integrated KED should be the same as the calculated kinetic energy
    assert_allclose(T, scf.energies.Ekin)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
