#!/usr/bin/env python3
"""Test total energies for bulk silicon with custom k-points."""

from numpy.testing import assert_allclose
import pytest

from eminus import Cell, RSCF, USCF
from eminus.kpoints import kpoint_convert

# Total energy from a spin-paired calculation with PWDFT.jl with the same parameters as below
E_ref = -7.493530216

a = 10.2631
ecut = 5
s = 15
guess = 'random'
etol = 1e-6
opt = {'auto': 27}
wk = [0.4, 0.6]
k = [[0.1, 0.2, 0.3], [0.1, 0.1, 0.1]]


def test_polarized():
    """Compare total energies for a test system with a reference value (spin-paired)."""
    cell = Cell('Si', 'diamond', ecut=ecut, a=a)
    cell.s = s
    cell.set_k(kpoint_convert(k, cell.a), wk)
    E = USCF(cell, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref, atol=etol)


def test_unpolarized():
    """Compare total energies for a test system with a reference value (spin-paired)."""
    cell = Cell('Si', 'diamond', ecut=ecut, a=a)
    cell.s = s
    cell.set_k(kpoint_convert(k, cell.a), wk)
    E = RSCF(cell, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref, atol=etol)


if __name__ == '__main__':
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)

# using PWDFT

# const psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pade_gth")
# const psp = [joinpath(psp_path, "Si-q4.gth")]

# atoms = Atoms(xyz_string_frac=
#     """
#     2

#     Si  0.0   0.0   0.0
#     Si  0.25  0.25  0.25
#     """,
#     in_bohr=true,
#     LatVecs=gen_lattice_fcc(10.2631))
# Ham = Hamiltonian(atoms, psp, 5.0, Nspin=1, xcfunc="VWN", kpts_str=
#     """
#     2
#     0.1 0.2 0.3 0.4
#     0.1 0.1 0.1 0.6
#     """)
# KS_solve_Emin_PCG!(Ham, startingrhoe=:random, verbose=false, etot_conv_thr=1e-6)
# println("VWN: $(round.(sum(Ham.energies); digits=6))")
