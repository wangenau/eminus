# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, overload

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

def solve_poisson(
    atoms: Atoms,
    n: NDArray[float64],
) -> NDArray[float64]: ...
@overload
def orth(
    atoms: Atoms,
    W: NDArray[complex128],
) -> NDArray[complex128]: ...
@overload
def orth(
    atoms: Atoms,
    W: list[NDArray[complex128]],
) -> list[NDArray[complex128]]: ...
@overload
def orth_unocc(
    atoms: Atoms,
    Y: NDArray[complex128],
    Z: NDArray[complex128],
) -> NDArray[complex128]: ...
@overload
def orth_unocc(
    atoms: Atoms,
    Y: list[NDArray[complex128]],
    Z: list[NDArray[complex128]],
) -> list[NDArray[complex128]]: ...
def get_n_total(
    atoms: Atoms,
    Y: list[NDArray[complex128]],
    n_spin: NDArray[float64] | None = ...,
) -> NDArray[float64]: ...
@overload
def get_n_spin(
    atoms: Atoms,
    Y: list[NDArray[complex128]],
) -> NDArray[float64]: ...
@overload
def get_n_spin(
    atoms: Atoms,
    Y: NDArray[complex128],
    ik: int,
) -> NDArray[float64]: ...
@overload
def get_n_single(
    atoms: Atoms,
    Y: list[NDArray[complex128]],
) -> NDArray[float64]: ...
@overload
def get_n_single(
    atoms: Atoms,
    Y: NDArray[complex128],
    ik: int,
) -> NDArray[float64]: ...
def get_grad(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[NDArray[complex128]],
    **kwargs: Any,
) -> NDArray[complex128]: ...
def H(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[NDArray[complex128]],
    dn_spin: NDArray[float64] | None = ...,
    phi: NDArray[float64] | None = ...,
    vxc: NDArray[complex128] | None = ...,
    vsigma: NDArray[complex128] | None = ...,
    vtau: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...
def H_precompute(
    scf: SCF,
    W: list[NDArray[complex128]],
) -> tuple[
    NDArray[float64],
    NDArray[complex128],
    NDArray[complex128],
    NDArray[complex128],
    NDArray[complex128],
]: ...
def Q(
    inp: NDArray[complex128],
    U: NDArray[complex128],
) -> NDArray[complex128]: ...
def get_psi(
    scf: SCF,
    W: list[NDArray[complex128]] | None,
    **kwargs: Any,
) -> list[NDArray[complex128]]: ...
def get_epsilon(
    scf: SCF,
    W: list[NDArray[complex128]] | None,
    **kwargs: Any,
) -> NDArray[float64]: ...
def get_epsilon_unocc(
    scf: SCF,
    W: list[NDArray[complex128]] | None,
    Z: list[NDArray[complex128]] | None,
    **kwargs: Any,
) -> NDArray[float64]: ...
def guess_random(
    scf: SCF,
    Nstate: int | None = ...,
    seed: int = ...,
    symmetric: bool = ...,
) -> list[NDArray[complex128]]: ...
def guess_pseudo(
    scf: SCF,
    Nstate: int | None = ...,
    seed: int = ...,
    symmetric: bool = ...,
) -> list[NDArray[complex128]]: ...
