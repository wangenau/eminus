import numpy as np
from numpy.linalg import inv
from numpy.random import randn
from scipy.linalg import sqrtm
from setup import *
from operators import *

dr = norm(r - np.sum(R, axis=1) / 2, axis=1)
sigma1 = 0.25
g1 = np.exp(-dr**2 / (2 * sigma1**2)) / np.sqrt(2 * np.pi * sigma1**2)**3
g1 = Z * (np.sum(g1) * det(R) / np.prod(S)) * g1
n = cI(cJ(g1) * Sf)
n = np.real(n)
phi = cI(Linv(-4 * np.pi * O(cJ(n))))
phi = np.real(phi)
Unum = 0.5 * np.real(cJ(phi).conj().T @ O(cJ(n)))
Uself = Z**2 / (2 * np.sqrt(np.pi)) * (1 / sigma1) * X.shape[0]

lamda = 18.5
rc = 1.052
Gm = np.sqrt(G2)
Vps = -2 * np.pi * np.exp(-np.pi * Gm / lamda) * np.cos(rc * Gm) * (Gm / lamda) / (1 - np.exp(-2 * np.pi * Gm / lamda))
for n in range(5):
    Vps = Vps + (-1)**n * np.exp(-lamda * rc * n) / (1 + (n * lamda / Gm)**2)
Vps = Vps * 4 * np.pi * Z / Gm**2 * (1 + np.exp(-lamda * rc)) - 4 * np.pi * Z / Gm**2
n = np.arange(1, 5)
Vps[0] = 4 * np.pi * Z * (1 + np.exp(-lamda * rc)) * (rc**2 / 2 + 1 / lamda**2 * (np.pi**2 / 6 + np.sum((-1)**n * np.exp(-lamda * rc * n) / n**2)))

Vdual = cJ(Vps * Sf)
W = randn(len(active[0]), 16) + 1j * randn(len(active[0]), 16)
W = orth(W)
W, Elist = sd(W, Vdual, 200)
W = orth(W)
W, Elist = pccg(W, Vdual, 200, 1)
print('Compression: %f (theoretical: %f)'% (len(G2) / len(G2c), 1 / (4 * np.pi * (1 / 4)**3 / 3)))
print('Ewald energy:', (Unum - Uself)[0][0])
print('Electronic energy:', Elist[-1])
print('Total energy:', (Elist[-1] + Unum - Uself)[0][0])
