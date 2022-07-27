from eminus import Atoms, SCF

# # Start by creating an `Atoms` object for helium
# # Use a very small `ecut` for a fast calculation
atoms = Atoms('He', [0, 0, 0], ecut=5, Nspin=1)

# # Optional parameters with examples are listed as follows
# # Dictionary to set the maximum amount of steps per minimization method and their order
# # Set it to a very small value for a short output
min_dict = {'pccg': 5}

# # The `SCF` class only needs an `Atoms` object, but only calculate 5 steps for less output
print('First calculation:')
SCF(atoms, min=min_dict).run()

# # Exchange-correlation functional description (case insensitive), separated by a comma
xc = 'lda,pw'

# # The LibXC interface can be used by adding `'libxc:'` before a functional
# # Names and numbers can be used, and mixed with the internal functionals as well
# xc = 'libxc:LDA_X,libxc:LDA_C_PW'
# xc = 'libxc:1,pw'

# # Type of pseudopotential (case insensitive)
pot = 'gth'

# # Initial guess method for the basis functions (case insensitive)
guess = 'random'

# # Convergence tolerance of the total energy
etol = 1e-8

# # Conjugated-gradient from for the preconditioned conjugate-gradient minimization (pccg)
cgform = 2

# # The amount of output can be controlled with the verbosity level
# # By default the verbosity level of the `Atoms` object will be used
verbose = 4

# # Start a new calculation with new parameters
print('\nSecond calculation with more output:')
scf = SCF(atoms=atoms, xc=xc, pot=pot, guess=guess, etol=etol, min=min_dict, cgform=cgform,
          verbose=verbose)
etot = scf.run()

# # The total energy is a return value of the `SCF.run` function, but it is saved in the `SCF` object as well with all energy contributions
print(f'\nEnergy from SCF function = {etot} Eh')
print(f'\nEnergy in Atoms object:\n{scf.energies}')
