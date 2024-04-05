# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
from numpy.typing import NDArray

def lda_c_vwn(
    n: NDArray[np.float64],
    A: float = ...,
    b: float = ...,
    c: float = ...,
    x0: float = ...,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None]: ...
def lda_c_vwn_spin(
    n: NDArray[np.float64],
    zeta: NDArray[np.float64],
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None]: ...
