# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test D3 dispersion corrections."""

import pytest

from eminus import Atoms, SCF
from eminus.energies import get_Edisp
from eminus.testing import assert_allclose

atoms = Atoms(
    "CH4",
    (
        (0, 0, 0),
        (1.186, 1.186, 1.186),
        (1.186, -1.186, -1.186),
        (-1.186, 1.186, -1.186),
        (-1.186, -1.186, 1.186),
    ),
    ecut=1,
)


@pytest.mark.parametrize("xc", ["svwn", "pbe", "pbesol", "chachiyo"])
def test_functionals(xc):
    """Test the use of different functionals in dispersion corrections."""
    pytest.importorskip("dftd3", reason="dftd3 not installed, skip tests")
    ref = {"svwn": 0.0184646051, "pbe": -0.0011935254, "pbesol": -0.0003419625}
    ref["chachiyo"] = ref["pbe"]
    scf = SCF(atoms, xc=xc)
    Edisp = get_Edisp(scf)
    assert_allclose(Edisp, ref[xc], atol=1e-8)
    assert isinstance(Edisp, float)

    # import ase
    # from dftd3.ase import DFTD3
    # import numpy as np
    # from eminus.units import bohr2ang, ev2ha
    # atm = ase.Atoms(atoms.atom, bohr2ang(atoms.pos))
    # atm.cell = bohr2ang(atoms.a)
    # atm.pbc = np.array([1, 1, 1])
    # atm.calc = DFTD3(method=xc, damping="d3bj", atm=True)
    # print(ev2ha(atm.get_potential_energy()))


@pytest.mark.parametrize("atm", [True, False])
@pytest.mark.parametrize("version", ["d3bj", "d3bjm", "d3zero", "d3zerom", "d3op"])
def test_versions(atm, version):
    """Test the use of different dispersion correction versions."""
    pytest.importorskip("dftd3", reason="dftd3 not installed, skip tests")
    scf = SCF(atoms, xc="pbe")
    Edisp = get_Edisp(scf, version=version, atm=atm)
    assert Edisp == scf.energies.Edisp


@pytest.mark.parametrize("disp", [True, False, {"version": "d3zero", "atm": False, "xc": "scan"}])
def test_scf(disp):
    """Test the use in the SCF class."""
    pytest.importorskip("dftd3", reason="dftd3 not installed, skip tests")
    scf = SCF(atoms, xc="pbe", etol=10, disp=disp)
    scf.run()
    assert scf.energies.Edisp <= 0


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
