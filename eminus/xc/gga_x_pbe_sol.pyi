# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import floating
from numpy.typing import NDArray

type _Float = floating[Any]
type _ArrayReal = NDArray[_Float]

def gga_x_pbe_sol(
    n: _ArrayReal,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
def gga_x_pbe_sol_spin(
    n: _ArrayReal,
    zeta: _ArrayReal,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
