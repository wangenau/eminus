#!/usr/bin/env python3
"""Test orbital functions."""
import glob
import os

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, RSCF
from eminus.orbitals import cube_writer, FLO, FO, KSO, WO

atoms = Atoms('He', (0, 0, 0), ecut=1, center=True).build()
scf = RSCF(atoms)
scf.run()


def test_kso():
    """Test the Kohn-Sham orbital function."""
    orb = KSO(scf, write_cubes=True)[0]
    os.remove('He_KSO_k0_0.cube')
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_fo():
    """Test the Fermi orbital function."""
    orb = FO(scf, write_cubes=True, fods=[atoms.pos])[0]
    os.remove('He_FO_k0_0.cube')
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_flo():
    """Test the Fermi-Loewdin orbital function."""
    orb = FLO(scf, write_cubes=True, fods=[atoms.pos])[0]
    os.remove('He_FLO_k0_0.cube')
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_wo():
    """Test the Wannier orbital function."""
    orb = WO(scf, write_cubes=True, precondition=False)[0]
    os.remove('He_WO_k0_0.cube')
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


@pytest.mark.parametrize('unrestricted', [True, False])
def test_cube_writer(unrestricted):
    """Test the orbital cube writer function."""
    atoms.unrestricted = unrestricted
    atoms.f = np.ones((1, 2 - unrestricted, 2))
    rng = default_rng()
    Iorbital = [rng.standard_normal((atoms.occ.Nspin, atoms.Ns, atoms.occ.Nstate))]
    cube_writer(atoms, 'TMP', Iorbital)
    for f in glob.glob('He_TMP_k0_*.cube'):
        os.remove(f)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
