import numpy as np
from numpy.linalg import norm, det
from setup import *
from operators import *

dr = norm(r - np.sum(R, axis=1) / 2, axis=1)
sigma1 = 0.25
g1 = np.exp(-dr**2 / (2 * sigma1**2)) / np.sqrt(2 * np.pi * sigma1**2)**3
g1 = Z * (np.sum(g1) * det(R) / np.prod(S)) * g1
n = cI(cJ(g1) * Sf)
n = np.real(n)
print('Normalization check on g1: %20.16f' % (np.sum(g1) * det(R) / np.prod(S)))
print('Total charge check: %20.16f' % (np.sum(n) * det(R) / np.prod(S)))
phi = cI(Linv(-4 * np.pi * O(cJ(n))))
phi = np.real(phi)
Unum = 0.5 * np.real(cJ(phi).conj().T @ O(cJ(n)))
Uself = Z**2 / (2 * np.sqrt(np.pi)) * (1 / sigma1) * X.shape[0]
print('Ewald energy: %20.16f' % (Unum - Uself))
