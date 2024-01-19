# dft_calculations

Test total energies for a small set of systems for spin-paired and spin-polarized DFT calculations for different functionals.

The geometries used are experimental data taken from [CCCBDB](https://cccbdb.nist.gov/introx.asp). The exact origin is stated inside the XYZ files.

The reference data can be created with the Julia package [PWDFT.jl](https://github.com/f-fathurrahman/PWDFT.jl) for the VWN calculations using

```terminal
# using Pkg
# Pkg.add(PackageSpec(url="https://github.com/f-fathurrahman/PWDFT.jl"))
using PWDFT

const systems = ["H", "H2", "He", "Li", "LiH", "CH4", "Ne"]
const psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pade_gth")
const psps = [
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

and for the PBE calculations

```terminal
using PWDFT

const systems = ["H", "H2", "He", "Li", "LiH", "CH4", "Ne"]
const psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pbe_gth")
const psps = [
    [joinpath(psp_path, "H-q1.gth")],
    [joinpath(psp_path, "H-q1.gth")],
    [joinpath(psp_path, "He-q2.gth")],
    [joinpath(psp_path, "Li-q3.gth")],
    [joinpath(psp_path, "Li-q3.gth"), joinpath(psp_path, "H-q1.gth")],
    [joinpath(psp_path, "C-q4.gth"), joinpath(psp_path, "H-q1.gth")],
    [joinpath(psp_path, "Ne-q8.gth")]
]

println("E_ref = {")
for i = 1:size(systems, 1)
    atoms = Atoms(xyz_file=systems[i]*".xyz", LatVecs=gen_lattice_sc(10.0))
    Ham = Hamiltonian(atoms, psps[i], 10.0, Nspin=2, xcfunc="PBE")
    KS_solve_Emin_PCG!(Ham, startingrhoe=:random, verbose=false, etot_conv_thr=1e-6)
    println("    '$(systems[i])': $(round.(sum(Ham.energies); digits=6)),")
end
println("}")
```

and for the SCAN calculations

```terminal
using PWDFT

const systems = ["CH4"]
const psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pbe_gth")
const psps = [[joinpath(psp_path, "C-q4.gth"), joinpath(psp_path, "H-q1.gth")]]

println("E_ref = {")
for i = 1:size(systems, 1)
    atoms = Atoms(xyz_file=systems[i]*".xyz", LatVecs=gen_lattice_sc(10.0))
    Ham = Hamiltonian(atoms, psps[i], 10.0, Nspin=1, xcfunc="SCAN")
    KS_solve_Emin_PCG!(Ham, startingrhoe=:random, verbose=false, etot_conv_thr=1e-6)
    println("    '$(systems[i])': $(round.(sum(Ham.energies); digits=6)),")
end
println("}")
```

and for the bulk calculations

```terminal
using PWDFT

const xcfuncs = ["VWN", "PBE", "SCAN"]

println("E_ref = {")
for i = 1:size(xcfuncs, 1)
    if xcfuncs[i] == "VWN"
        psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pade_gth")
        psp = [joinpath(psp_path, "Si-q4.gth")]
    else
        psp_path = joinpath(dirname(pathof(PWDFT)), "..", "pseudopotentials", "pbe_gth")
        psp = [joinpath(psp_path, "Si-q4.gth")]
    end

    atoms = Atoms(xyz_string_frac=
        """
        2

        Si  0.0   0.0   0.0
        Si  0.25  0.25  0.25
        """,
        in_bohr=true,
        LatVecs=gen_lattice_fcc(10.2631))
    Ham = Hamiltonian(atoms, psp, 5.0, Nspin=1, xcfunc=xcfuncs[i], meshk=[2,2,2])
    KS_solve_Emin_PCG!(Ham, startingrhoe=:random, verbose=false, etot_conv_thr=1e-6)
    println("    '$(xcfuncs[i])': $(round.(sum(Ham.energies); digits=6)),")
end
println("}")
```

Some calculations use [JDFTx](https://jdftx.org) for calculating reference values. The pseudopotentials for the calculation have been taken from [QE](https://pseudopotentials.quantum-espresso.org/legacy_tables/hartwigesen-goedecker-hutter-pp), where the input file, e.g, for the charged systems looks like

```terminal
ion He 0 0 0 1
lattice 10 0 0 0 10 0 0 0 10
coords-type cartesian
elec-ex-corr lda-VWN
wavefunction random
elec-cutoff 10
ion-species He.pz-hgh.UPF
spintype z-spin
elec-initial-charge -charge
elec-initial-magnetization charge%2 yes
dump End None
```
