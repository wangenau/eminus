# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from numpy import float64, int64
from numpy.typing import NDArray

type Array1D = Sequence[float] | NDArray[float64]
type Array2D = Sequence[Sequence[float]] | Sequence[NDArray[float64]] | NDArray[float64]
type Array3D = (
    Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[NDArray[float64]]] | NDArray[float64]
)
type IntArray = Sequence[int] | NDArray[int64]
