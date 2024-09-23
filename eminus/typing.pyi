# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import TypeAlias

from numpy import float64, int64
from numpy.typing import NDArray

Array1D: TypeAlias = Sequence[float] | NDArray[float64]
Array2D: TypeAlias = Sequence[Sequence[float]] | Sequence[NDArray[float64]] | NDArray[float64]
Array3D: TypeAlias = (
    Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[NDArray[float64]]] | NDArray[float64]
)
IntArray: TypeAlias = Sequence[int] | NDArray[int64]
