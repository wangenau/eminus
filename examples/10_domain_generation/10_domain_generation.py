from timeit import default_timer

from eminus import Atoms, read_xyz, SCF
from eminus.addons import KSO, view_grid
from eminus.domains import domain_cuboid, domain_isovalue, domain_sphere, truncate
from eminus.energies import get_Esic, get_n_single

# Start by creating an Atoms object for lithium hydride
# Use a small s to make the resulting grid not too dense to display it
atoms = Atoms(*read_xyz('LiH.xyz'), s=30, center=True)
SCF(atoms)

# Create a boolean mask for a cuboidal domain
# This will create a domain with side lengths of 3 Bohr,
# with the center in the center at the center of mass of our molecule
mask = domain_cuboid(atoms, 3)

# Display the domain along with the atom positions
# The view_grid function can be used outside of notebooks
view_grid(atoms.r[mask], atoms.X)

# The same can be done for a spherical domain with a radius of 3 Bohr
mask = domain_sphere(atoms, 3)
view_grid(atoms.r[mask], atoms.X)

# One can also define more than one center
# This will create multiple domains and merge them, here shown with the atom positions as centers
mask = domain_sphere(atoms, 3, atoms.X)
view_grid(atoms.r[mask], atoms.X)

# An isovalue can be used to generate a domain from a real-space field data like orbitals
psi = KSO(atoms)
mask = domain_isovalue(psi[:, 0], 1e-2)
view_grid(atoms.r[mask], atoms.X)

# The same can be done for the density
# mask = domain_isovalue(atoms.n, 1e-3)
# view_grid(atoms.r[mask], atoms.X)

# Truncated densities can be used to calculate, e.g., SIC energies
# Calculate the single-electron densities from Kohn-Sham orbitals first
ni = get_n_single(atoms, atoms.J(psi))

# Calculate the SIC energy for untruncated densities
start = default_timer()
esic = get_Esic(atoms, atoms.W, ni)
end = default_timer()
print(f'Esic(untruncated) = {esic:.9f} Eh\nTime(untruncated) =  {end - start:.6f} s')

# Calculate the SIC energy for truncated densities
# One can notice a small energy deviation, but a faster calculation time
mask = domain_isovalue(ni, 1e-4)
ni_trunc = truncate(ni, mask)
start = default_timer()
esic_trunc = get_Esic(atoms, atoms.W, ni_trunc)
end = default_timer()
print(f'Esic( truncated ) = {esic_trunc:.9f} Eh\nTime( truncated ) =  {end - start:.6f} s')

# The truncated SIC energy will converge for smaller isovalues to the untruncated value
# mask = domain_isovalue(ni, 0)
# ni_trunc = truncate(ni, mask)
# esic_trunc = get_Esic(atoms, atoms.W, ni_trunc)
# print(esic == esic_trunc)
