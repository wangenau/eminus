#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test total energies for bulk silicon for different functionals."""

from numpy.testing import assert_allclose
import pytest

from eminus import Cell, RSCF, USCF

# Total energies from a spin-paired calculation with PWDFT.jl with the same parameters as below
# PWDFT.jl does not support spin-polarized calculations with SCAN
E_ref = {
    'SVWN': -7.785143,
    'PBE': -7.726629,
    ':MGGA_X_SCAN,:MGGA_C_SCAN': -7.729585,
}

a = 10.2631
ecut = 5
s = 15
kmesh = 2
guess = 'random'
etol = 1e-6
opt = {'auto': 25}


@pytest.mark.slow()
@pytest.mark.parametrize('xc', E_ref.keys())
def test_polarized(xc):
    """Compare total energies for a test system with a reference value (spin-paired)."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    cell = Cell('Si', 'diamond', ecut=ecut, a=a, kmesh=kmesh)
    cell.s = s
    E = USCF(cell, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[xc], rtol=etol)  # Use rtol over atol so SCAN can pass the test


@pytest.mark.parametrize('xc', E_ref.keys())
def test_unpolarized(xc):
    """Compare total energies for a test system with a reference value (spin-paired)."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    cell = Cell('Si', 'diamond', ecut=ecut, a=a, kmesh=kmesh)
    cell.s = s
    E = RSCF(cell, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[xc], rtol=etol)  # Use rtol over atol so SCAN can pass the test


if __name__ == '__main__':
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)

# using PWDFT

# const xcfuncs = ["VWN", "PBE", "SCAN"]

# println("E_ref = {")
# for i = 1:size(xcfuncs, 1)
#     if xcfuncs[i] == "VWN"
#         psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pade_gth")
#         psp = [joinpath(psp_path, "Si-q4.gth")]
#     else
#         psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pbe_gth")
#         psp = [joinpath(psp_path, "Si-q4.gth")]
#     end

#     atoms = Atoms(xyz_string_frac=
#         """
#         2

#         Si  0.0   0.0   0.0
#         Si  0.25  0.25  0.25
#         """,
#         in_bohr=true,
#         LatVecs=gen_lattice_fcc(10.2631))
#     Ham = Hamiltonian(atoms, psp, 5.0, Nspin=1, xcfunc=xcfuncs[i], meshk=[2,2,2])
#     KS_solve_Emin_PCG!(Ham, startingrhoe=:random, verbose=false, etot_conv_thr=1e-6)
#     println("    '$(xcfuncs[i])': $(round.(sum(Ham.energies); digits=6)),")
# end
# println("}")
