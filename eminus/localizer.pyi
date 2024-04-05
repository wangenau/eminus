# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import overload, Sequence

import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .typing import Array1D, Array2D

@overload
def eval_psi(
    atoms: Atoms,
    psi: NDArray[np.complex128],
    r: Array1D | Array2D,
) -> NDArray[np.complex128]: ...
@overload
def eval_psi(
    atoms: Atoms,
    psi: list[NDArray[np.complex128]],
    r: Array1D | Array2D,
) -> list[NDArray[np.complex128]]: ...
@overload
def get_R(
    atoms: Atoms,
    psi: NDArray[np.complex128],
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> NDArray[np.complex128]: ...
@overload
def get_R(
    atoms: Atoms,
    psi: list[NDArray[np.complex128]],
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> list[NDArray[np.complex128]]: ...
@overload
def get_FO(
    atoms: Atoms,
    psi: NDArray[np.complex128],
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> NDArray[np.complex128]: ...
@overload
def get_FO(
    atoms: Atoms,
    psi: list[NDArray[np.complex128]],
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> list[NDArray[np.complex128]]: ...
def get_S(
    atoms: Atoms,
    psirs: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
@overload
def get_FLO(
    atoms: Atoms,
    psi: NDArray[np.complex128],
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> NDArray[np.complex128]: ...
@overload
def get_FLO(
    atoms: Atoms,
    psi: list[NDArray[np.complex128]],
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> list[NDArray[np.complex128]]: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: NDArray[np.complex128],
) -> NDArray[np.float64]: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: list[NDArray[np.complex128]],
) -> list[NDArray[np.float64]]: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: NDArray[np.complex128],
) -> NDArray[np.float64]: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: list[NDArray[np.complex128]],
) -> list[NDArray[np.float64]]: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: NDArray[np.complex128],
) -> NDArray[np.float64]: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: list[NDArray[np.complex128]],
) -> list[NDArray[np.float64]]: ...
def wannier_supercell_matrices(
    atoms: Atoms,
    psirs: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def wannier_supercell_cost(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
) -> float: ...
def wannier_supercell_grad(
    atoms: Atoms,
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
) -> NDArray[np.complex128]: ...
@overload
def get_wannier(
    atoms: Atoms,
    psirs: NDArray[np.complex128],
    Nit: int = ...,
    conv_tol: float = ...,
    mu: float = ...,
    random_guess: bool = ...,
    seed: int | None = ...,
) -> NDArray[np.complex128]: ...
@overload
def get_wannier(
    atoms: Atoms,
    psirs: list[NDArray[np.complex128]],
    Nit: int = ...,
    conv_tol: float = ...,
    mu: float = ...,
    random_guess: bool = ...,
    seed: int | None = ...,
) -> list[NDArray[np.complex128]]: ...
