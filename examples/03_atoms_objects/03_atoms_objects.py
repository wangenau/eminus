from eminus import Atoms

# # The only necessary parameters are `atom` and `pos`
# # `atom` holds the atom symbols, and `pos` holds the atom positions
# # Please note that atomic units will be used
atom = 'N2'
pos = [[0, 0, 0], [2.074, 0, 0]]

# # Create an object for dinitrogen and display it
atoms = Atoms(atom, pos)
print(f'Atoms object:\n{atoms}\n')

# # Cut-off energy
ecut = 20

# # Optional parameters with examples are listed as follows
# # Cell size or vacuum size
a = 20

# # Spin of the system, i.e., the number of unpaired electrons (2S, not 2S+1)
spin = 0

# # Total charge of the system
charge = 0

# # Spin handling
unrestricted = False

# # Center the system inside the box by its geometric center of mass and rotate it such that its geometric moment of inertia aligns with the coordinate axes
center = True

# # Level of output, larger numbers mean more output
verbose = 4

# # Create an `Atoms` object for dinitrogen and display it
atoms = Atoms(atom=atom, pos=pos, ecut=ecut, a=a, spin=spin, charge=charge,
              unrestricted=unrestricted, center=center, verbose=verbose)
print(f'New Atoms object:\n{atoms}\n')

# # Albeit discouraged, some properties can be changed after the initialization as seen below
# # Valence charge per atom, the charges should not differ for the same species
# # `None` will use valence charges from GTH pseudopotentials
atoms.Z = [5, 5]

# # Real-space sampling of the cell using an equidistant grid
atoms.s = 40

# # Occupation numbers per state
# # `None` will assume occupations of 2
# # The last state will be adjusted if the sum of `f` is not equal to the sum of `Z`
atoms.f = [2, 2, 2, 2, 2]

# # You can manipulate the object by displaying or editing most properties
# # To display the calculated cell volume
print(f'Cell volume = {atoms.Omega} a0^3')

# # If you edit properties of an existing object dependent properties can be updated by rebuilding the `Atoms` object
# # The `atoms.build` function is used to generate cell parameters for an SCF calculation, but an SCF object will call the function if necessary
# # Edit the cell size, rebuild the object, and display the new cell volume
atoms.a = 3
atoms.build()
print(f'New cell volume = {atoms.Omega} a0^3')

# # To display all relevant electronic information regarding the occupations, one can display the `Occupations` object every `Atoms` object has
print(f'\nOccupations object:\n{atoms.occ}')

# # More information is always available in the respective docstring
# print(f'\nAtoms docstring:\n{Atoms.__doc__}')
# # or:
# help(Atoms)
