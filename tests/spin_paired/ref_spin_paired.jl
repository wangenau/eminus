using PWDFT


function ref_spin_paired()
    path = "../../plainedft/pade_gth/"
    systems = ["H", "H2", "LiH", "CH4", "Ne"]
    a = 16.0
    ecut = 10.0
    psps = [
        [joinpath(path, "H-q1.gth")],
        [joinpath(path, "H-q1.gth")],
        [joinpath(path, "Li-q1.gth"), joinpath(path, "H-q1.gth")],
        [joinpath(path, "C-q4.gth"), joinpath(path, "H-q1.gth")],
        [joinpath(path, "Ne-q8.gth")]
    ]

    Etots_ref = Vector{Float64}()
    for i = 1:size(systems, 1)
        atoms = Atoms(xyz_file=systems[i] * ".xyz", LatVecs=gen_lattice_sc(a))
        Ham = Hamiltonian(atoms, psps[i], ecut)
        KS_solve_Emin_PCG!(Ham, etot_conv_thr=1e-7)
        append!(Etots_ref, sum(Ham.energies))
    end

    println(round.(Etots_ref; digits=8))
end


ref_spin_paired()
