#!/usr/bin/env python3
"""Test total energies for bulk lithium (spin-polarized) for different smearings."""
from numpy.testing import assert_allclose
import pytest

from eminus import Cell, USCF

# Total energies from a spin-polarized calculation with PWDFT.jl with similar parameters as below
E_ref = {
    1e-4: -5.404468,
    1e-3: -5.404468
}

a = 3.44
ecut = 10
s = 9
xc = 'LDA,VWN'
guess = 'random'
etol = 1e-6
opt = {'sd': 12, 'pccg': 16}
betat = 1e-3


@pytest.mark.parametrize('smearing', E_ref.keys())
def test_polarized(smearing):
    """Compare total energies for a test system with a reference value (spin-polarized)."""
    cell = Cell('Li', 'bcc', ecut=ecut, a=a, smearing=smearing)
    cell.s = s
    E = USCF(cell, xc=xc, guess=guess, etol=etol, opt=opt).run(betat=betat)
    assert_allclose(E, E_ref[smearing], atol=etol)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)

# using PWDFT

# const smearing = [1e-4, 1e-3]
# const psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pade_gth")
# const psp = [joinpath(psp_path, "Li-q3.gth")]

# println("E_ref = {")
# for i = 1:size(smearing, 1)
#     atoms = Atoms(xyz_string_frac=
#         """
#         1

#         Li  0.0   0.0   0.0
#         """,
#         in_bohr=true,
#         LatVecs=gen_lattice_bcc(3.44))
#     Ham = Hamiltonian(atoms, psp, 10.0, Nspin=2, xcfunc="VWN")
#     KS_solve_SCF!(Ham, startingrhoe=:random, verbose=false, betamix=1.0, etot_conv_thr=1e-6,
#                   use_smearing=true, kT=smearing[i])
#     println("    $(smearing[i]): $(round.(sum(Ham.energies); digits=6)),")
# end
# println("}")
