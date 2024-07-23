# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from numpy import float64
from numpy.typing import NDArray

_DnOrNone = TypeVar('_DnOrNone', NDArray[float64], None)
_TauOrNone = TypeVar('_TauOrNone', NDArray[float64], None)

# Create a custom Callable type for functionals
class _FunctionalType(Protocol):
    def __call__(
        self,
        n: NDArray[float64],
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[NDArray[float64], NDArray[float64], None]
        | tuple[NDArray[float64], NDArray[float64], NDArray[float64]]
    ): ...

def get_xc(
    xc: str | Sequence[str],
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    dens_threshold: float = ...,
) -> tuple[NDArray[float64], NDArray[float64], _DnOrNone, _TauOrNone]: ...
def get_exc(
    xc: str | Sequence[str],
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    dens_threshold: float = ...,
) -> NDArray[float64]: ...
def get_vxc(
    xc: str | Sequence[str],
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    dens_threshold: float = ...,
) -> tuple[NDArray[float64], _DnOrNone, _TauOrNone]: ...
def parse_functionals(xc: str) -> list[str]: ...
def parse_xc_type(xc: str) -> str: ...
def parse_xc_libxc(xc_id: int | str) -> str: ...
def parse_xc_pyscf(xc_id: int | str) -> str: ...
def get_zeta(n_spin: NDArray[float64]) -> NDArray[float64]: ...
def mock_xc(
    n: NDArray[float64],
    Nspin: int = ...,
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], None]: ...

IMPLEMENTED: dict[str, _FunctionalType]
XC_MAP: dict[str, str]
ALIAS: dict[str, str]
