# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TypeAlias

from numpy import floating
from numpy.typing import NDArray

_Float: TypeAlias = floating[Any]
_ArrayReal: TypeAlias = NDArray[_Float]

def gga_x_pbe(
    n: _ArrayReal,
    mu: float = ...,
    dn_spin: _ArrayReal | None = ...,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
def gga_x_pbe_spin(
    n: _ArrayReal,
    zeta: _ArrayReal,
    mu: float = ...,
    dn_spin: _ArrayReal | None = ...,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
def pbe_x_base(
    n: _ArrayReal,
    mu: float = ...,
    dn: _ArrayReal | None = ...,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
