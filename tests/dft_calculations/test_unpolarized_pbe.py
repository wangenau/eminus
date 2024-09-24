# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test total energies using PBE for a small set of spin-paired systems."""

import inspect
import pathlib

import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, read, RSCF

# Total energies from a spin-polarized calculation with PWDFT.jl with the same parameters as below
# Closed-shell systems have the same energy for spin-paired and -polarized calculations
E_ref = {
    "H2": -1.131175,
    "He": -2.588705,
    "LiH": -6.580019,
    "CH4": -7.745197,
    "Ne": -29.965354,
}


@pytest.mark.parametrize("system", E_ref.keys())
def test_unpolarized(system):
    """Compare total energies for a test system with a reference value (spin-paired)."""
    file_path = pathlib.Path(inspect.stack()[0][1]).parent
    a = 10
    ecut = 10
    s = 30
    xc = "pbe"
    guess = "random"
    etol = 1e-6
    opt = {"auto": 25}

    atom, X = read(str(file_path.joinpath(f"{system}.xyz")))
    atoms = Atoms(atom, X, a=a, ecut=ecut)
    atoms.s = s
    E = RSCF(atoms, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[system], atol=etol)


if __name__ == "__main__":
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
