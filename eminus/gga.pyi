# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

def get_grad_field(
    atoms: Atoms,
    field: NDArray[floating] | NDArray[complexfloating],
    real: bool = ...,
) -> NDArray[floating] | NDArray[complexfloating]: ...
def gradient_correction(
    atoms: Atoms,
    spin: int,
    dn_spin: NDArray[floating],
    vsigma: NDArray[complexfloating],
) -> NDArray[complexfloating]: ...
@overload
def get_tau(
    atoms: Atoms,
    Y: NDArray[complexfloating],
    ik: int,
) -> NDArray[floating]: ...
@overload
def get_tau(
    atoms: Atoms,
    Y: list[NDArray[complexfloating]],
) -> NDArray[floating]: ...
def calc_Vtau(
    scf: SCF,
    ik: int,
    spin: int,
    W: NDArray[complexfloating],
    vtau: NDArray[complexfloating],
) -> NDArray[complexfloating]: ...
