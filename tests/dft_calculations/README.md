# dft_calculations

Test total energies for a small set of systems for spin-paired and spin-polarized DFT calculations.

The geometries used are experimental data taken from [CCCBDB](https://cccbdb.nist.gov/introx.asp). The exact origin is stated inside the xyz files.

The reference data can be created with the Julia package [PWDFT.jl](https://github.com/f-fathurrahman/PWDFT.jl) using

```bash
# using Pkg
# Pkg.add(PackageSpec(url="https://github.com/f-fathurrahman/PWDFT.jl"))
using PWDFT

systems = ["H", "H2", "He", "Li", "LiH", "CH4", "Ne"]
psp_path = "../../eminus/pade/"
psps = [
    [joinpath(psp_path, "H-q1.gth")],
    [joinpath(psp_path, "H-q1.gth")],
    [joinpath(psp_path, "He-q2.gth")],
    [joinpath(psp_path, "Li-q1.gth")],
    [joinpath(psp_path, "Li-q1.gth"), joinpath(psp_path, "H-q1.gth")],
    [joinpath(psp_path, "C-q4.gth"), joinpath(psp_path, "H-q1.gth")],
    [joinpath(psp_path, "Ne-q8.gth")]
]

println("E_ref = {")
for i = 1:size(systems, 1)
    atoms = Atoms(xyz_file=systems[i]*".xyz", LatVecs=gen_lattice_sc(10.0))
    Ham = Hamiltonian(atoms, psps[i], 10.0, Nspin=2, xcfunc="VWN")
    KS_solve_Emin_PCG!(Ham, startingrhoe=:random, verbose=false, etot_conv_thr=1e-6)
    println("    '$(systems[i])': $(round.(sum(Ham.energies); digits=6)),")
end
println("}")
```
