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

atoms = Atoms('He', (0, 0, 0), center=True).build()
atoms.s = 10
scf = RSCF(atoms)
scf.run()
dV = atoms.Omega / np.prod(atoms.s)


def test_kso():
    """Test the Kohn-Sham orbital function."""
    orb = KSO(scf, write_cubes=True)
    os.remove('He_KSO_0.cube')
    assert_allclose(dV * np.sum(orb.conj() * orb), 1)


def test_fo():
    """Test the Fermi orbital function."""
    orb = FO(scf, write_cubes=True, fods=[atoms.X])
    os.remove('He_FO_0.cube')
    assert_allclose(dV * np.sum(orb.conj() * orb), 1)


def test_flo():
    """Test the Fermi-Loewdin orbital function."""
    orb = FLO(scf, write_cubes=True, fods=[atoms.X])
    os.remove('He_FLO_0.cube')
    assert_allclose(dV * np.sum(orb.conj() * orb), 1)


def test_wo():
    """Test the Wannier orbital function."""
    orb = WO(scf, write_cubes=True, precondition=False)
    os.remove('He_WO_0.cube')
    assert_allclose(dV * np.sum(orb.conj() * orb), 1)


@pytest.mark.parametrize('unrestricted', [True, False])
def test_cube_writer(unrestricted):
    """Test the orbital cube writer function."""
    atoms.unrestricted = unrestricted
    atoms.f = np.array([[1], [1]])
    rng = default_rng()
    orbital = rng.standard_normal((atoms.occ.Nspin, len(atoms.G2c), atoms.occ.Nstate))
    cube_writer(atoms, 'TMP', orbital)
    for f in glob.glob('He_TMP_0*.cube'):
        os.remove(f)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
