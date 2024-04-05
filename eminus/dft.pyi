# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

def solve_poisson(
    atoms: Atoms,
    n: NDArray[np.float64],
) -> NDArray[np.float64]: ...
@overload
def orth(
    atoms: Atoms,
    W: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
@overload
def orth(
    atoms: Atoms,
    W: list[NDArray[np.complex128]],
) -> list[NDArray[np.complex128]]: ...
@overload
def orth_unocc(
    atoms: Atoms,
    Y: NDArray[np.complex128],
    Z: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
@overload
def orth_unocc(
    atoms: Atoms,
    Y: list[NDArray[np.complex128]],
    Z: list[NDArray[np.complex128]],
) -> list[NDArray[np.complex128]]: ...
def get_n_total(
    atoms: Atoms,
    Y: list[NDArray[np.complex128]],
    n_spin: NDArray[np.float64] | None = ...,
) -> NDArray[np.float64]: ...
@overload
def get_n_spin(
    atoms: Atoms,
    Y: list[NDArray[np.complex128]],
) -> NDArray[np.float64]: ...
@overload
def get_n_spin(
    atoms: Atoms,
    Y: NDArray[np.complex128],
    ik: int,
) -> NDArray[np.float64]: ...
@overload
def get_n_single(
    atoms: Atoms,
    Y: list[NDArray[np.complex128]],
) -> NDArray[np.float64]: ...
@overload
def get_n_single(
    atoms: Atoms,
    Y: NDArray[np.complex128],
    ik: int,
) -> NDArray[np.float64]: ...
def get_grad(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[NDArray[np.complex128]],
    **kwargs: Any,
) -> NDArray[np.complex128]: ...
def H(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[NDArray[np.complex128]],
    dn_spin: NDArray[np.float64] | None = ...,
    phi: NDArray[np.float64] | None = ...,
    vxc: NDArray[np.complex128] | None = ...,
    vsigma: NDArray[np.complex128] | None = ...,
    vtau: NDArray[np.complex128] | None = ...,
) -> NDArray[np.complex128]: ...
def H_precompute(
    scf: SCF,
    W: list[NDArray[np.complex128]],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]: ...
def Q(
    inp: NDArray[np.complex128],
    U: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
def get_psi(
    scf: SCF,
    W: list[NDArray[np.complex128]] | None,
    **kwargs: Any,
) -> list[NDArray[np.complex128]]: ...
def get_epsilon(
    scf: SCF,
    W: list[NDArray[np.complex128]] | None,
    **kwargs: Any,
) -> NDArray[np.float64]: ...
def get_epsilon_unocc(
    scf: SCF,
    W: list[NDArray[np.complex128]] | None,
    Z: list[NDArray[np.complex128]] | None,
    **kwargs: Any,
) -> NDArray[np.float64]: ...
def guess_random(
    scf: SCF,
    Nstate: int | None = ...,
    seed: int = ...,
    symmetric: bool = ...,
) -> list[NDArray[np.complex128]]: ...
def guess_pseudo(
    scf: SCF,
    Nstate: int | None = ...,
    seed: int = ...,
    symmetric: bool = ...,
) -> list[NDArray[np.complex128]]: ...
