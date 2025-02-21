# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test orbital functions."""

import copy
import os
import pathlib

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose

from eminus import Atoms, RSCF
from eminus.orbitals import cube_writer, FLO, FO, KSO, SCDM, WO

atoms = Atoms("He", (0, 0, 0), ecut=1, center=True).build()
scf = RSCF(atoms)
scf.run()


def test_kso():
    """Test the Kohn-Sham orbital function."""
    scf_tmp = copy.deepcopy(scf)
    scf_tmp.kpts.Nk = 2
    scf_tmp.kpts.path = "GX"
    scf_tmp.atoms.build()
    assert scf.W is not None
    scf_tmp.W = scf_tmp.atoms.J([atoms.I(scf.W)[0], atoms.I(scf.W)[0]], full=False)
    orb = KSO(scf_tmp, write_cubes=True)[0]
    os.remove("He_KSO_k0_0.cube")
    os.remove("He_KSO_k1_0.cube")
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_fo():
    """Test the Fermi orbital function."""
    orb = FO(scf, write_cubes=True, fods=[atoms.pos])[0]
    os.remove("He_FO_k0_0.cube")
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_fo_from_wannier():
    """Test the Fermi orbital function starting from COMs of Wannier orbitals."""
    orb = FO(scf, write_cubes=True)[0]
    os.remove("He_FO_k0_0.cube")
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_fo_from_pycom():
    """Test the Fermi orbital function starting from a PyCOM guess calculated with PySCF."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    orb = FO(scf, write_cubes=True, guess="pycom")[0]
    os.remove("He_FO_k0_0.cube")
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_flo():
    """Test the Fermi-Loewdin orbital function."""
    orb = FLO(scf, write_cubes=True, fods=[atoms.pos])[0]
    os.remove("He_FLO_k0_0.cube")
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_flo_from_wannier():
    """Test the Fermi-Loewdin orbital function starting from COMs of Wannier orbitals."""
    orb = FLO(scf, write_cubes=True)[0]
    os.remove("He_FLO_k0_0.cube")
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_flo_from_pycom():
    """Test the Fermi-Loewdin orbital function starting from a PyCOM guess calculated with PySCF."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    orb = FLO(scf, write_cubes=True, guess="pycom")[0]
    os.remove("He_FLO_k0_0.cube")
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


@pytest.mark.parametrize("precondition", [True, False])
def test_wo(precondition):
    """Test the Wannier orbital function."""
    orb = WO(scf, write_cubes=True, precondition=precondition)[0]
    os.remove("He_WO_k0_0.cube")
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


def test_scdm():
    """Test the SCDM orbital function."""
    orb = SCDM(scf, write_cubes=True)[0]
    os.remove("He_SCDM_k0_0.cube")
    assert_allclose(atoms.dV * np.sum(orb.conj() * orb), 1)


@pytest.mark.parametrize("unrestricted", [True, False])
def test_cube_writer(unrestricted):
    """Test the orbital cube writer function."""
    atoms.unrestricted = unrestricted
    atoms.f = np.ones((1, 2 - unrestricted, 2))
    rng = default_rng()
    Iorbital = [rng.standard_normal((atoms.occ.Nspin, atoms.Ns, atoms.occ.Nstate))]
    cube_writer(atoms, "TMP", Iorbital)
    for f in pathlib.Path().glob("He_TMP_k0_*.cube"):
        os.remove(f)


if __name__ == "__main__":
    import inspect

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
