import numpy as np
from numpy.linalg import inv
from numpy.random import randn
from setup import *
from operators import *

dr = norm(r - np.sum(R, axis=1) / 2, axis=1)
V = 2 * dr**2
Vdual = cJdag(O(cJ(V)))
W = randn(np.prod(S), 4) + 1j * randn(np.prod(S), 4)
W = orth(W)
W = sd(W, Vdual, 400)
print('Total energy:', getE(W, Vdual))
Psi, epsilon = getPsi(W, Vdual)
print(epsilon)
