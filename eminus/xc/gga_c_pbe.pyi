# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TypeAlias

from numpy import floating
from numpy.typing import NDArray

_Float: TypeAlias = floating
_ArrayReal: TypeAlias = NDArray[_Float]

def gga_c_pbe(
    n: _ArrayReal,
    beta: float = ...,
    dn_spin: _ArrayReal | None = ...,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
def gga_c_pbe_spin(
    n: _ArrayReal,
    zeta: _ArrayReal,
    beta: float = ...,
    dn_spin: _ArrayReal | None = ...,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
