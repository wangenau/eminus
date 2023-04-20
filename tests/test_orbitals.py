#!/usr/bin/env python3
'''Test orbital functions.'''
import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, RSCF
from eminus.orbitals import FLO, FO, KSO, WO

atoms = Atoms('He', (0, 0, 0), s=10, center=True)
scf = RSCF(atoms, min={'pccg': 10})
scf.run()
dV = atoms.Omega / np.prod(atoms.s)


def test_kso():
    '''Test the Kohn-Sham orbital function.'''
    orb = KSO(scf, write_cubes=True)
    os.remove('He_KSO_0.cube')
    assert_allclose(dV * np.sum(orb.conj() * orb), 1)


def test_fo():
    '''Test the Fermi orbital function.'''
    orb = FO(scf, write_cubes=True, fods=[atoms.X])
    os.remove('He_FO_0.cube')
    assert_allclose(dV * np.sum(orb.conj() * orb), 1)


def test_flo():
    '''Test the Fermi-Löwdin orbital function.'''
    orb = FLO(scf, write_cubes=True, fods=[atoms.X])
    os.remove('He_FLO_0.cube')
    assert_allclose(dV * np.sum(orb.conj() * orb), 1)


def test_wo():
    '''Test the Wannier orbital function.'''
    orb = WO(scf, write_cubes=True, precondition=False)
    os.remove('He_WO_0.cube')
    assert_allclose(dV * np.sum(orb.conj() * orb), 1)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
