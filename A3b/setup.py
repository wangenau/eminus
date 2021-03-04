import numpy as np
from numpy.linalg import norm, inv

S = np.array([48, 48, 48])
a = 5.66 / 0.52917721
R = a * np.eye(3)
X = a * np.array([[0, 0, 0], [0.25, 0.25, 0.25], [0, 0.5, 0.5], [0.25, 0.75, 0.75], [0.5, 0, 0.5], [0.75, 0.25, 0.25], [0.5, 0.5, 0], [0.75, 0.75, 0.25]])
Z = 4
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
if any((S % 2) != 0):
    print('Odd dimension in S, this is really bad!')
eS = S / 2 + 0.5
edges = np.nonzero(np.any(np.abs(M - np.ones((np.size(M, axis=0), 1)) @ [eS]) < 1, axis=1))
G2mx = np.min(G2[edges])
active = np.nonzero(G2 < G2mx / 4)
G2c = G2[active]
