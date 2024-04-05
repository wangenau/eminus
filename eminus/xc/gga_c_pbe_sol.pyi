# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
from numpy.typing import NDArray

def gga_c_pbe_sol(
    n: NDArray[np.float64],
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def gga_c_pbe_sol_spin(
    n: NDArray[np.float64],
    zeta: NDArray[np.float64],
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
