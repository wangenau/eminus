# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

@overload
def get_grad_field(
    atoms: Atoms,
    field: NDArray[float64],
    real: bool = ...,
) -> NDArray[float64]: ...
@overload
def get_grad_field(
    atoms: Atoms,
    field: NDArray[complex128],
    real: bool = ...,
) -> NDArray[float64] | NDArray[complex128]: ...
def gradient_correction(
    atoms: Atoms,
    spin: int,
    dn_spin: NDArray[float64],
    vsigma: NDArray[complex128],
) -> NDArray[complex128]: ...
@overload
def get_tau(
    atoms: Atoms,
    Y: NDArray[complex128],
    ik: int,
) -> NDArray[float64]: ...
@overload
def get_tau(
    atoms: Atoms,
    Y: list[NDArray[complex128]],
) -> NDArray[float64]: ...
def calc_Vtau(
    scf: SCF,
    ik: int,
    spin: int,
    W: NDArray[complex128],
    vtau: NDArray[complex128],
) -> NDArray[complex128]: ...
