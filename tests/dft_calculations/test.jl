using Base
using LinearAlgebra
using Printf
using PWDFT

const common_path = joinpath(dirname(pathof(PWDFT)), "..", "examples", "common")
include(common_path * "/gen_kpath.jl")
const psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pade_gth")
const psp = [joinpath(psp_path, "Si-q4.gth")]

atoms = Atoms(xyz_string_frac=
    """
    2

    Si  0.0  0.0  0.0
    Si  0.25  0.25  0.25
    """,
    in_bohr=true,
    LatVecs=gen_lattice_fcc(10.2631))

const ecutwfc = 5.0
const Nspin = 1
Ham = Hamiltonian(atoms, psp, ecutwfc, Nspin=Nspin, xcfunc="VWN", meshk=[2,2,2])
KS_solve_Emin_PCG!(Ham, startingrhoe=:random, verbose=false, etot_conv_thr=1e-6)

# Do this to not use non-ASCII characters
const Deltak = Dict(Base.kwarg_decl.(methods(gen_kpath))[1][1] => 0.4)
kpoints, kpt_spec, kpt_spec_labels = gen_kpath(atoms, "L-G-X", "fcc"; Deltak...)
Ham.pw = PWGrid(ecutwfc, atoms.LatVecs, kpoints=kpoints)
const Nkpt = Ham.pw.gvecw.kpoints.Nkpt
Ham.electrons = Electrons(atoms, Ham.pspots, Nspin=Nspin, Nkpt=kpoints.Nkpt, Nstates_empty=4)
Ham.pspotNL = PsPotNL(atoms, Ham.pw, Ham.pspots)
psiks = rand_BlochWavefunc(Ham)
evals = zeros(Float64, Ham.electrons.Nstates, Nkpt * Nspin)
for ispin = 1:Nspin
    for ik = 1:Nkpt
        Ham.ik = ik
        Ham.ispin = ispin
        ikspin = ik + (ispin - 1) * Nkpt
        evals[:, ikspin], psiks[ikspin] = diag_LOBPCG(Ham, psiks[ikspin], verbose_last=true)
    end
end
