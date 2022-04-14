from eminus import Atoms, SCF

# Start by creating an Atoms object for helium
# Use a very small ecut for a fast calculation
atoms = Atoms('He', [0, 0, 0], ecut=5)

# Optional parameters with examples are listed as follows
# Dictionary to set the maximum amount of steps per minimization method and their order
# Set it to a very small value for a short output
min = {'pccg': 5}

# The SCF function only needs an Atoms object, but only calculate 5 steps for less output
print('First calculation:')
SCF(atoms, min=min)

# Initial guess method for the basis functions (case insensitive)
guess = 'random'

# Convergence tolerance of the total energy
etol = 1e-8

# Conjugated-gradient from for the preconditioned conjugate-gradient minimization (pccg)
cgform = 2

# The amount of output can be controlled with the verbosity level
atoms.verbose = 5

# Start a new calculation with new parameters
print('\nSecond calculation with more output:')
etot = SCF(atoms=atoms, guess=guess, etol=etol, min=min, cgform=cgform)

# The total energy is a return value of the SCF function, but it is saved in the Atoms object as
# well with all energy contributions
print(f'\nEnergy from SCF function = {etot} Eh')
print(f'\nEnergy in Atoms object:\n{atoms.energies}')
