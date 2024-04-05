# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array1D: TypeAlias = Sequence[float] | NDArray[np.float64]
Array2D: TypeAlias = Sequence[Sequence[float]] | Sequence[NDArray[np.float64]] | NDArray[np.float64]
Array3D: TypeAlias = (
    Sequence[Sequence[Sequence[float]]]
    | Sequence[Sequence[NDArray[np.float64]]]
    | NDArray[np.float64]
)
IntArray: TypeAlias = Sequence[int] | NDArray[np.int64]
