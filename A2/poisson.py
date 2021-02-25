import numpy as np
from numpy.linalg import norm, det
from setup import *
from operators import *

# Code to solve Poisson’s equation
# Compute distances dr to center point in cell
dr = norm(r - np.sum(R, axis=1) / 2, axis=1)
# Compute two normalized Gaussians (widths 0.50 and 0.75)
sigma1 = 0.75
g1 = np.exp(-dr**2 / (2 * sigma1**2)) / np.sqrt(2 * np.pi * sigma1**2)**3
sigma2 = 0.5
g2 = np.exp(-dr**2 / (2 * sigma2**2)) / np.sqrt(2 * np.pi * sigma2**2)**3
# Define charge density as the difference
n = g2 - g1
# Check norms and integral (should be near 1 and 0, respectively)
print('Normalization check on g1: %20.16f' % (np.sum(g1) * det(R) / np.prod(S)))
print('Normalization check on g2: %20.16f' % (np.sum(g2) * det(R) / np.prod(S)))
print('Total charge check: %20.16f' % (np.sum(n) * det(R) / np.prod(S)))
# Solve Poisson’s equation
phi = cI(Linv(-4 * np.pi * O(cJ(n))))
# Due to rounding, tiny imaginary parts creep into the solution. Eliminate
# by taking the real part.
phi = np.real(phi)
# Check total Coulomb energy
Unum = 0.5 * np.real(cJ(phi).conj().T @ O(cJ(n)))
Uanal = ((1 / sigma1 + 1 / sigma2) / 2 - np.sqrt(2) / np.sqrt(sigma1**2 + sigma2**2)) / np.sqrt(np.pi)
print('Numeric, analytic Coulomb energy: %20.16f, %20.16f' % (Unum, Uanal))
