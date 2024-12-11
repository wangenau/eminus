# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any, Protocol

from numpy import complexfloating
from numpy.typing import NDArray

from .minimizer import _Conditionype
from .scf import SCF

# Create a custom Callable type for cost functions
class _CostType(Protocol):
    def __call__(
        self,
        scf: SCF,
        W: list[NDArray[complexfloating]],
    ) -> float: ...

# Create a custom Callable type for gradient functions
class _GradType(Protocol):
    def __call__(
        self,
        scf: SCF,
        ik: int,
        spin: int,
        W: list[NDArray[complexfloating]],
        **kwargs: Any,
    ) -> NDArray[complexfloating]: ...

def scf_step_occ(
    scf: SCF,
    W: list[NDArray[complexfloating]],
) -> float: ...
def scf_step_unocc(
    scf: SCF,
    Z: list[NDArray[complexfloating]],
) -> float: ...
def get_grad_occ(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[NDArray[complexfloating]],
    **kwargs: Any,
) -> NDArray[complexfloating]: ...
def get_grad_unocc(
    scf: SCF,
    ik: int,
    spin: int,
    Z: list[NDArray[complexfloating]],
    **kwargs: Any,
) -> NDArray[complexfloating]: ...
def sd(
    scf: SCF,
    W: list[NDArray[complexfloating]],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    **kwargs: Any,
) -> tuple[list[float], list[NDArray[complexfloating]]]: ...
def pclm(
    scf: SCF,
    W: list[NDArray[complexfloating]],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    precondition: bool = ...,
    **kwargs: Any,
) -> tuple[list[float], list[NDArray[complexfloating]]]: ...
def lm(
    scf: SCF,
    W: list[NDArray[complexfloating]],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    **kwargs: Any,
) -> tuple[list[float], list[NDArray[complexfloating]]]: ...
def pccg(
    scf: SCF,
    W: list[NDArray[complexfloating]],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
    precondition: bool = ...,
) -> tuple[list[float], list[NDArray[complexfloating]]]: ...
def cg(
    scf: SCF,
    W: list[NDArray[complexfloating]],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
) -> tuple[list[float], list[NDArray[complexfloating]]]: ...
def auto(
    scf: SCF,
    W: list[NDArray[complexfloating]],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
) -> tuple[list[float], list[NDArray[complexfloating]]]: ...

IMPLEMENTED: dict[str, Callable[[Any], tuple[list[float], NDArray[complexfloating]]]]
