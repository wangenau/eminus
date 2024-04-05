# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import overload

import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

@overload
def get_grad_field(
    atoms: Atoms,
    field: NDArray[np.float64],
    real: bool = ...,
) -> NDArray[np.float64]: ...
@overload
def get_grad_field(
    atoms: Atoms,
    field: NDArray[np.complex128],
    real: bool = ...,
) -> NDArray[np.float64] | NDArray[np.complex128]: ...
def gradient_correction(
    atoms: Atoms,
    spin: int,
    dn_spin: NDArray[np.float64],
    vsigma: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
@overload
def get_tau(
    atoms: Atoms,
    Y: NDArray[np.complex128],
    ik: int,
) -> NDArray[np.float64]: ...
@overload
def get_tau(
    atoms: Atoms,
    Y: list[NDArray[np.complex128]],
) -> NDArray[np.float64]: ...
def calc_Vtau(
    scf: SCF,
    ik: int,
    spin: int,
    W: NDArray[np.complex128],
    vtau: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
