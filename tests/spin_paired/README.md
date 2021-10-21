# spin_paired

Test total energies for a small set of systems.

The geometries used are experimental data taken from [CCCBDB](https://cccbdb.nist.gov/introx.asp). The exact origin is stated inside the xyz files.

The file `ref_spin_paired.jl` was used to create reference data with [PWDFT.jl](https://github.com/f-fathurrahman/PWDFT.jl).

It can be executed with Julia using

```bash
julia ref_spin_paired.jl
```

