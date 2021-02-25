import numpy as np
from numpy.linalg import norm, inv

#S = np.array([3, 3, 3])
S = np.array([20, 25, 30])
R = np.diag([6, 6, 6])
X = np.array([[0, 0, 0], [1.75, 0, 0]])
Z = 1
ms = np.arange(0, np.prod(S))
m1 = ms % S[0]
m2 = np.floor(ms / S[0]) % S[1]
m3 = np.floor(ms / (S[0] * S[1])) % S[2]
M = np.array([m1, m2, m3]).T
n1 = m1 - (m1 > S[0] / 2) * S[0]
n2 = m2 - (m2 > S[1] / 2) * S[1]
n3 = m3 - (m3 > S[2] / 2) * S[2]
N = np.array([n1, n2, n3]).T
r = M @ inv(np.diag(S)) @ R.T
G = 2 * np.pi * N @ inv(R)
G2 = np.sum(G**2, axis=1)
Sf = np.sum(np.exp(-1j * G @ X.conj().T), axis=1)
f = 2
