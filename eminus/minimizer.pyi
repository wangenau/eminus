# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

# Create a custom Callable type for cost functions
class CostType(Protocol):
    def __call__(
        self,
        scf: SCF,
        step: int,
    ) -> float: ...

# Create a custom Callable type for gradient functions
class GradType(Protocol):
    def __call__(
        self,
        scf: SCF,
        ik: int,
        spin: int,
        W: list[NDArray[np.complex128]],
        **kwargs: Any,
    ) -> NDArray[np.complex128]: ...

# Create a custom Callable type for condition functions
class Conditionype(Protocol):
    def __call__(
        self,
        scf: SCF,
        method: str,
        Elist: list[float],
        linmin: NDArray[np.float64] | None = ...,
        cg: NDArray[np.float64] | None = ...,
        norm_g: NDArray[np.float64] | None = ...,
    ) -> bool: ...

def scf_step(
    scf: SCF,
    step: int,
) -> float: ...
def check_convergence(
    scf: SCF,
    method: str,
    Elist: list[float],
    linmin: NDArray[np.float64] | None = ...,
    cg: NDArray[np.float64] | None = ...,
    norm_g: NDArray[np.float64] | None = ...,
) -> bool: ...
def print_scf_step(
    scf: SCF,
    method: str,
    Elist: list[float],
    linmin: NDArray[np.float64] | None,
    cg: NDArray[np.float64] | None,
    norm_g: NDArray[np.float64] | None,
) -> None: ...
def linmin_test(
    g: NDArray[np.complex128],
    d: NDArray[np.complex128],
) -> float: ...
def cg_test(
    atoms: Atoms,
    ik: int,
    g: NDArray[np.complex128],
    g_old: NDArray[np.complex128],
    precondition: bool = ...,
) -> float: ...
def cg_method(
    scf: SCF,
    ik: int,
    cgform: int,
    g: NDArray[np.complex128],
    g_old: NDArray[np.complex128],
    d_old: NDArray[np.complex128],
    precondition: bool = ...,
) -> tuple[float, float]: ...
def sd(
    scf: SCF,
    Nit: int,
    cost: CostType = ...,
    grad: GradType = ...,
    condition: Conditionype = ...,
    betat: float = ...,
    **kwargs: Any,
) -> list[float]: ...
def pclm(
    scf: SCF,
    Nit: int,
    cost: CostType = ...,
    grad: GradType = ...,
    condition: Conditionype = ...,
    betat: float = ...,
    precondition: bool = ...,
    **kwargs: Any,
) -> list[float]: ...
def lm(
    scf: SCF,
    Nit: int,
    cost: CostType = ...,
    grad: GradType = ...,
    condition: Conditionype = ...,
    betat: float = ...,
    **kwargs: Any,
) -> list[float]: ...
def pccg(
    scf: SCF,
    Nit: int,
    cost: CostType = ...,
    grad: GradType = ...,
    condition: Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
    precondition: bool = ...,
) -> list[float]: ...
def cg(
    scf: SCF,
    Nit: int,
    cost: CostType = ...,
    grad: GradType = ...,
    condition: Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
) -> list[float]: ...
def auto(
    scf: SCF,
    Nit: int,
    cost: CostType = ...,
    grad: GradType = ...,
    condition: Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
) -> list[float]: ...

IMPLEMENTED: dict[str, Callable[[Any], list[float]]]
