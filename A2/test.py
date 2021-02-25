import numpy as np
from numpy.linalg import norm, det
from setup import *
from operators import *
from numpy.random import randn
a=randn(np.prod(S))+1j*randn(np.prod(S))
b=randn(np.prod(S))+1j*randn(np.prod(S))
a=np.array([a,a])
b=np.array([b,b])
print(cIdag(a))
print(cJdag(a))
print(cIdag(b))
print(cJdag(b))
