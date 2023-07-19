import numpy as np
from scipy.optimize import curve_fit

from eminus import Atoms, SCF
from eminus.units import ry2ha
import eminus.xc

# # Start with a normal SCF calculation for helium with the LDA Chachiyo functional
atoms = Atoms('He', (0, 0, 0), ecut=10, verbose=0)
scf = SCF(atoms, xc='lda,chachiyo')
scf.run()
print(f'Energies with the Chachiyo functional:\n{scf.energies}')

# # Build a new custom functional by fitting the Chachiyo functional over the Quantum Monte Carlo results from Ceperley and Alder
# # The values are the paramagnetic correlation energies from Tab. 5 in Can. J. Phys. 58, 1200
rs = np.array([2, 5, 10, 20, 50, 100])
# Convert from -mRy to Hartree
ec = -ry2ha(np.array([90.2, 56.3, 37.22, 23.00, 11.40, 6.379])) / 1000

# # Define the original Chachiyo functional parameters
# # The parameters are derived such that the functional recovers the values in Phys. Rev. B 84, 033103 for the high-density limit
a = (np.log(2) - 1) / (2 * np.pi**2)
b = 20.4562557
print(f'\nOriginal parameter:\nb = {b:.7f}')

# # All functionals in eminus share the same signature and return types
# # Define a custom spin-paired functional using said signature based on the Chachiyo functional but with the customizable parameter parameter `b` so we can fit over it
def custom_functional(n, b, **kwargs):
    rs = (3 / (4 * np.pi * n))**(1 / 3)  # Wigner-Seitz radius
    ec = a * np.log(1 + b / rs + b / rs**2)  # Exchange energy
    vc = ec + a * b * (2 + rs) / (3 * (b + b * rs + rs**2))  # Exchange potential
    return ec, np.array([vc]), None

# # Fit the functional using a wrapper function
# # We only fit over the energies so the wrapper function only returns the first argument of the functional
# # The third return value of the functional (here `None`) is only used in GGA functionals
# # Functionals usually take densities as the input, so convert the Wigner-Seitz radius to a density in the wrapper function
def functional_wrapper(rs, b):
    n = 3 / (4 * np.pi * rs**3)  # Density from Wigner-Seitz radius
    return custom_functional(n, b)[0]

# # Do the fit over the Quantum Monte Carlo data
# # The resulting parameters won't recover the high-density limit but can be more accurate in different density regimes
fitted_b, _ = curve_fit(functional_wrapper, rs, ec, p0=b)
print(f'\nFitted parameter:\nb = {fitted_b[0]:.7f}')

# # Some modules of eminus have an IMPLEMENTED lookup dictionary for functions that can be extended
# # These are available in the xc, potentials, and minimizer modules and can be used as seen below
# # Plug the fitted parameters into the custom functional
eminus.xc.IMPLEMENTED['custom'] = lambda n, **kwargs: custom_functional(n, *fitted_b)

# # Start an SCF calculation with our custom functional
scf = SCF(atoms, xc='lda,custom')
scf.run()
print(f'\nEnergies with the fitted Chachiyo functional:\n{scf.energies}')

# # Something similar has been done by Karasiev in J. Chem. Phys. 145, 157101 but for a functional of the form `a * np.log(1 + b / rs + c / rs**2)` without the restriction of `b=c`
# # Nonetheless, our fitted value of `b=21.7914970` is very similar to the one derived by Karasiev with `b=21.7392245`
