from eminus import Atoms

# The only necessary parameters are atom and X
# atom holds the atom symbols, and X holds the atom positions
# Please note that atomic units will be used
atom = 'N2'
X = [[0, 0, 0], [2.074, 0, 0]]

# Create an object for dinitrogen and display it
atoms = Atoms(atom, X)
print(f'Atoms object:\n{atoms}\n')

# Optional parameters with examples are listed as follows
# Cell size or vacuum size
a = 20

# Cut-off energy
ecut = 20

# Valence charge per atom. The charges should not differ for the same species.
# None will use valence charges from GTH pseudopotentials
Z = [5, 5]

# Real-space sampling of the cell using an equidistant grid
S = 40

# Occupation numbers per state
# None will assume occupations of 2. The last state will be adjusted if the sum of f is not equal to
# the sum of Z.
f = [2, 2, 2, 2, 2]

# Number of states
# None will get the number of states from f or assume occupations of 2
Ns = 5

# Level of output. Larger numbers mean more output.
verbose = 4

# Type of pseudopotential (case insensitive).
pot = 'gth'

# Center the system inside the box by its geometric center of mass and rotate it such that its
# geometric moment of inertia aligns with the coordinate axes.
center = True

# Exchange-correlation functional description (case insensitive), separated by a comma.
exc = 'lda,pw'

# Create an object for dinitrogen and display it
atoms = Atoms(atom=atom, X=X, a=a, ecut=ecut, Z=Z, S=S, f=f, Ns=Ns, verbose=verbose,
              pot=pot, center=center, exc=exc)
print(f'New Atoms object:\n{atoms}\n')

# You can always manipulate the object freely by displaying or editing properties
# To display the calculated cell volume
print(f'Cell volume = {atoms.CellVol} a0^3')

# If you edit properties of an existing object dependent properties can be updated
# Edit the cell size, update the object, and display the new cell volume
atoms.a = 3
atoms.update()
print(f'New cell volume = {atoms.CellVol} a0^3')

# More informations are always available in the respective docstring
# print(f'\nAtoms docstring:\n{Atoms.__doc__}')
# or:
# help(Atoms)
