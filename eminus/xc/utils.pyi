# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from numpy import floating
from numpy.typing import NDArray

_DnOrNone = TypeVar("_DnOrNone", NDArray[floating], None)
_TauOrNone = TypeVar("_TauOrNone", NDArray[floating], None)

# Create a custom Callable type for functionals
class _FunctionalType(Protocol):
    def __call__(
        self,
        n: NDArray[floating],
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[NDArray[floating], NDArray[floating], None]
        | tuple[NDArray[floating], NDArray[floating], NDArray[floating]]
    ): ...

def get_xc(
    xc: str | Sequence[str],
    n_spin: NDArray[floating],
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, Any] | None = ...,
    dens_threshold: float = ...,
) -> tuple[NDArray[floating], NDArray[floating], _DnOrNone, _TauOrNone]: ...
def get_exc(
    xc: str | Sequence[str],
    n_spin: NDArray[floating],
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, Any] | None = ...,
    dens_threshold: float = ...,
) -> NDArray[floating]: ...
def get_vxc(
    xc: str | Sequence[str],
    n_spin: NDArray[floating],
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, Any] | None = ...,
    dens_threshold: float = ...,
) -> tuple[NDArray[floating], _DnOrNone, _TauOrNone]: ...
def parse_functionals(xc: str) -> list[str]: ...
def parse_xc_type(xc: str) -> str: ...
def parse_xc_libxc(xc_id: int | str) -> str: ...
def parse_xc_pyscf(xc_id: int | str) -> str: ...
def get_xc_defaults(xc: str | Sequence[str]) -> dict[str, Any]: ...
def get_zeta(n_spin: NDArray[floating]) -> NDArray[floating]: ...
def mock_xc(
    n: NDArray[floating],
    Nspin: int = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], None]: ...

IMPLEMENTED: dict[str, _FunctionalType]
XC_MAP: dict[str, str]
ALIAS: dict[str, str]
