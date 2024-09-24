# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test eigenenergies for bulk silicon in band structure calculations."""

import numpy as np
from numpy.testing import assert_allclose

from eminus import Cell, RSCF, USCF
from eminus.dft import get_epsilon, get_epsilon_unocc
from eminus.tools import get_bandgap

# Eigenenergies from a spin-paired calculation with PWDFT.jl with the same parameters as below
epsilon_ref = np.array(
    [
        [
            -0.1120735697,
            -0.0223086429,
            0.1874096636,
            0.1874096657,
            0.3024433520,
            0.3533744734,
            0.3533746945,
            0.5029139877,
        ],
        [
            -0.1991130675,
            0.2336390609,
            0.2336390610,
            0.2336390730,
            0.3236558181,
            0.3236558199,
            0.3236562389,
            0.3723404989,
        ],
        [
            -0.0471149116,
            -0.0471146587,
            0.1244731242,
            0.1244731242,
            0.2552557610,
            0.2552561594,
            0.5979018826,
            0.5979073606,
        ],
    ]
)
bandgap_ref = 0.021616688

a = 10.2631
ecut = 5
s = 15
kmesh = 2
guess = "sym-random"
etol = 1e-6
path = "LGX"


def test_polarized():
    """Compare band energies for a test system with reference values (spin-polarized)."""
    opt = {"auto": 27}

    cell = Cell("Si", "diamond", ecut=ecut, a=a, kmesh=kmesh, bands=8)
    cell.s = s
    scf = USCF(cell, guess=guess, etol=etol, opt=opt)
    scf.run()

    scf.kpts.path = path
    scf.kpts.Nk = len(path)
    scf.converge_bands()

    assert hasattr(scf, "_precomputed")
    epsilon_occ = get_epsilon(scf, scf.W, **scf._precomputed)
    epsilon_unocc = get_epsilon_unocc(scf, scf.W, scf.Z, **scf._precomputed)
    epsilon = np.append(epsilon_occ, epsilon_unocc, axis=2)
    # Eigenenergies are a bit more sensitive than total energies
    assert_allclose(epsilon[:, 0], epsilon_ref, atol=1e-5)
    assert_allclose(epsilon[:, 1], epsilon_ref, atol=1e-5)
    bandgap = get_bandgap(scf)
    assert_allclose(bandgap, bandgap_ref, atol=etol)


def test_unpolarized():
    """Compare band energies for a test system with reference values (spin-paired)."""
    opt = {"auto": 26}

    cell = Cell("Si", "diamond", ecut=ecut, a=a, kmesh=kmesh, bands=8)
    cell.s = s
    scf = RSCF(cell, guess=guess, etol=etol, opt=opt)
    scf.run()

    scf.kpts.path = path
    scf.kpts.Nk = len(path)
    scf.converge_bands()

    assert hasattr(scf, "_precomputed")
    epsilon_occ = get_epsilon(scf, scf.W, **scf._precomputed)
    epsilon_unocc = get_epsilon_unocc(scf, scf.W, scf.Z, **scf._precomputed)
    epsilon = np.append(epsilon_occ, epsilon_unocc, axis=2)
    # Eigenenergies are a bit more sensitive than total energies
    assert_allclose(epsilon[:, 0], epsilon_ref, atol=1e-5)
    bandgap = get_bandgap(scf)
    assert_allclose(bandgap, bandgap_ref, atol=etol)


if __name__ == "__main__":
    import inspect
    import pathlib

    import pytest

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)

# using Base
# using LinearAlgebra
# using Printf
# using PWDFT

# const common_path = joinpath(dirname(pathof(PWDFT)), "..", "examples", "common")
# include(common_path * "/gen_kpath.jl")
# const psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pade_gth")
# const psp = [joinpath(psp_path, "Si-q4.gth")]

# atoms = Atoms(xyz_string_frac=
#     """
#     2

#     Si  0.0  0.0  0.0
#     Si  0.25  0.25  0.25
#     """,
#     in_bohr=true,
#     LatVecs=gen_lattice_fcc(10.2631))

# const ecutwfc = 5.0
# const Nspin = 1
# Ham = Hamiltonian(atoms, psp, ecutwfc, Nspin=Nspin, xcfunc="VWN", meshk=[2,2,2])
# KS_solve_Emin_PCG!(Ham, startingrhoe=:random, verbose=false, etot_conv_thr=1e-6)

# # Do this to not use non-ASCII characters
# const Deltak = Dict(Base.kwarg_decl.(methods(gen_kpath))[1][1] => 0.4)
# kpoints, kpt_spec, kpt_spec_labels = gen_kpath(atoms, "L-G-X", "fcc"; Deltak...)
# Ham.pw = PWGrid(ecutwfc, atoms.LatVecs, kpoints=kpoints)
# const Nkpt = Ham.pw.gvecw.kpoints.Nkpt
# Ham.electrons = Electrons(atoms, Ham.pspots, Nspin=Nspin, Nkpt=kpoints.Nkpt, Nstates_empty=4)
# Ham.pspotNL = PsPotNL(atoms, Ham.pw, Ham.pspots)
# psiks = rand_BlochWavefunc(Ham)
# evals = zeros(Float64, Ham.electrons.Nstates, Nkpt * Nspin)
# for ispin = 1:Nspin
#     for ik = 1:Nkpt
#         Ham.ik = ik
#         Ham.ispin = ispin
#         ikspin = ik + (ispin - 1) * Nkpt
#         evals[:, ikspin], psiks[ikspin] = diag_LOBPCG(Ham, psiks[ikspin], verbose_last=true)
#     end
# end
