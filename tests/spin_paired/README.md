# spin_paired

Test total energies for a small set of systems.

The geometries used are experimental data taken from [CCCBDB](https://cccbdb.nist.gov/introx.asp). The exact origin is stated inside the xyz files.

The reference data can be created with the Julia package [PWDFT.jl](https://github.com/f-fathurrahman/PWDFT.jl) using

```bash
# using Pkg
# Pkg.add(PackageSpec(url="https://github.com/f-fathurrahman/PWDFT.jl"))
using PWDFT

path = "../../eminus/pade_gth/"
systems = ["He", "H2", "LiH", "CH4", "Ne"]
a = 16.0
ecut = 10.0
psps = [
    [joinpath(path, "He-q2.gth")],
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
```
