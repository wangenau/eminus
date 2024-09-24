# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test total energies for methane using SCAN."""

import inspect
import pathlib

import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, read, RSCF, USCF

# Total energies from a spin-paired calculation with PWDFT.jl with the same parameters as below
# PWDFT.jl does not support spin-polarized calculations with SCAN
# Only test methane for faster tests, also SCAN can easily run into convergence issues
E_ref = {
    "CH4": -7.75275,
}

file_path = pathlib.Path(inspect.stack()[0][1]).parent
a = 10
ecut = 10
s = 30
xc = ":MGGA_X_SCAN,:MGGA_C_SCAN"
guess = "random"
etol = 1e-6
opt = {"auto": 19}


@pytest.mark.parametrize("system", E_ref.keys())
def test_polarized(system):
    """Compare total energies for a test system with a reference value (spin-polarized)."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    atom, X = read(str(file_path.joinpath(f"{system}.xyz")))
    atoms = Atoms(atom, X, a=a, ecut=ecut)
    atoms.s = s
    E = USCF(atoms, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[system], atol=etol)


@pytest.mark.parametrize("system", E_ref.keys())
def test_unpolarized(system):
    """Compare total energies for a test system with a reference value (spin-paired)."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    atom, X = read(str(file_path.joinpath(f"{system}.xyz")))
    atoms = Atoms(atom, X, a=a, ecut=ecut)
    atoms.s = s
    E = RSCF(atoms, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[system], atol=etol)


if __name__ == "__main__":
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)

# using PWDFT

# const systems = ["CH4"]
# const psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pbe_gth")
# const psps = [[joinpath(psp_path, "C-q4.gth"), joinpath(psp_path, "H-q1.gth")]]

# println("E_ref = {")
# for i = 1:size(systems, 1)
#     atoms = Atoms(xyz_file=systems[i]*".xyz", LatVecs=gen_lattice_sc(10.0))
#     Ham = Hamiltonian(atoms, psps[i], 10.0, Nspin=1, xcfunc="SCAN")
#     KS_solve_Emin_PCG!(Ham, startingrhoe=:random, verbose=false, etot_conv_thr=1e-6)
#     println("    '$(systems[i])': $(round.(sum(Ham.energies); digits=6)),")
# end
# println("}")
