using LinearAlgebra
using Random
using Printf
using PWDFT
using PyCall

const DIR_PWDFT = joinpath(dirname(pathof(PWDFT)),"..")
const DIR_PSP = joinpath(DIR_PWDFT, "pseudopotentials", "pade_gth")
#const DIR_STRUCTURES = joinpath(DIR_PWDFT, "structures")

# Needed to build a wave function, bc I am too stupid to get the correct shape
include("/home/wanja/PWDFT.jl/src/gen_wavefunc.jl")

function main(name)
    Random.seed!(1234)

    if name == "Ne"
        # Atoms
        atoms = Atoms( xyz_file=joinpath("", "Ne.xyz"),
                       LatVecs = gen_lattice_sc(16.0) )

        # Initialize Hamiltonian
        pspfiles = [joinpath(DIR_PSP, "Ne-q8.gth")]
        ecutwfc = 25.0
        Ham = Hamiltonian( atoms, pspfiles, ecutwfc )

        # Create a basis set with a set seed, same as in python
        py"""
        from numpy.random import rand, seed
        seed(1234)
        # Has to have the same form as the python version (Number of G-vectors, States)
        xs = rand(24405, 4)
        """
        xs = PyArray(py"xs"o)
    end

    if name == "LiH"
        # Atoms
        atoms = Atoms( xyz_file=joinpath("", "LiH.xyz"),
                       LatVecs = gen_lattice_sc(16.0) )

        # Initialize Hamiltonian
        pspfiles = [joinpath(DIR_PSP, "Li-q1.gth"), joinpath(DIR_PSP, "H-q1.gth")]
        ecutwfc = 25.0
        Ham = Hamiltonian( atoms, pspfiles, ecutwfc )

        # Create a basis set with a set seed, same as in python
        py"""
        from numpy.random import rand, seed
        seed(1234)
        # Has to have the same form as the python version (Number of G-vectors, States)
        xs = rand(24405, 1)
        """
        xs = PyArray(py"xs"o)
    end

    # display(Ham)
    # Create wfc (just to get the correct shape)
    psiks = rand_wfc( Ham )
    psiks[1] = xs
    # psiks[1] = ortho_sqrt(psiks[1])

    E = calc_energies( Ham, psiks )
    V = op_V_Ps_nloc( Ham, psiks )

    betaNL_psi = calc_betaNL_psi( 1, Ham.pspotNL.betaNL, psiks[1] )

    return Ham, E.Ps_nloc, V, psiks[1], betaNL_psi
end

main("LiH")
