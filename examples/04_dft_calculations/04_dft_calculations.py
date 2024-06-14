# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from eminus import Atoms, SCF

# # Start by creating an `Atoms` object for helium
# # Use a very small `ecut` for a fast calculation
atoms = Atoms('He', [0, 0, 0], ecut=5)

# # Optional parameters with examples are listed as follows
# # Dictionary to set the maximum amount of steps per minimization method and their order
# # Set it to a very small value for a short output
opt = {'pccg': 5}

# # The `SCF` class only needs an `Atoms` object, but only calculate 5 steps for less output
print('First calculation:')
SCF(atoms, opt=opt).run()

# # Exchange-correlation functional description (case insensitive), separated by a comma
xc = 'lda,pw'

# # The Libxc interface can be used by adding `libxc:` before a functional (you can also neglect the `libxc` part)
# # Names and numbers can be used, and mixed with the internal functionals as well
# xc = 'libxc:LDA_X,:LDA_C_PW'
# xc = 'libxc:1,pw'

# # Type of potential (case insensitive)
pot = 'gth'

# # Initial guess method for the basis functions (case insensitive)
# # Adding `sym` to the string will use the same guess for all spin channels
# # An unsymmetric guess will be used by default
guess = 'random'

# # Convergence tolerance of the total energy
etol = 1e-8

# # Convergence tolerance of the gradient norm
gradtol = 1e-7

# # Calculate a self-interaction correction from the Kohn-Sham orbitals after an SCF calculation.
sic = True

# # Calculate a dispersion correction after an SCF calculation (need the dispersion extra).
disp = False

# # The amount of output can be controlled with the verbosity level
# # By default the verbosity level of the `Atoms` object will be used
verbose = 4

# # Start a new calculation with new parameters
print('\nSecond calculation with more output:')
scf = SCF(
    atoms=atoms,
    xc=xc,
    pot=pot,
    guess=guess,
    etol=etol,
    gradtol=gradtol,
    opt=opt,
    sic=sic,
    disp=disp,
    verbose=verbose,
)

# # Arguments for the minimizer can be passed through via the run function, e.g., for the conjugated-gradient form
etot = scf.run(cgform=2)

# # The total energy is a return value of the `SCF.run` function, but it is saved in the `SCF` object as well with all energy contributions
print(f'\nEnergy from SCF function = {etot} Eh')
print(f'\nEnergy in Atoms object:\n{scf.energies}')
