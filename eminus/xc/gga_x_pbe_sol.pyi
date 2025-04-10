# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TypeAlias

from numpy import floating
from numpy.typing import NDArray

_Float: TypeAlias = floating[Any]
_ArrayReal: TypeAlias = NDArray[_Float]

def gga_x_pbe_sol(
    n: _ArrayReal,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
def gga_x_pbe_sol_spin(
    n: _ArrayReal,
    zeta: _ArrayReal,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
