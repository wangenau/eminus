# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .scf import SCF

def harmonic(scf: SCF) -> NDArray[np.complex128]: ...
def coulomb(scf: SCF) -> NDArray[np.complex128]: ...
def ge(scf: SCF) -> NDArray[np.complex128]: ...
def init_pot(scf: SCF) -> NDArray[np.complex128]: ...

IMPLEMENTED: dict[str, Callable[[SCF], NDArray[np.complex128]]]
