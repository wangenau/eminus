# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
from numpy.typing import NDArray

def lda_c_chachiyo_mod(
    n: NDArray[np.float64],
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None]: ...
def chachiyo_scaling_mod(
    zeta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def lda_c_chachiyo_mod_spin(
    n: NDArray[np.float64],
    zeta: NDArray[np.float64],
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None]: ...
