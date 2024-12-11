# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import floating
from numpy.typing import NDArray

def gga_c_pbe_sol(
    n: NDArray[floating],
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...
def gga_c_pbe_sol_spin(
    n: NDArray[floating],
    zeta: NDArray[floating],
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...
