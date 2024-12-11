# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from numpy import floating, integer
from numpy.typing import NDArray

type Array1D = Sequence[float] | NDArray[floating]
type Array2D = Sequence[Sequence[float]] | Sequence[NDArray[floating]] | NDArray[floating]
type Array3D = (
    Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[NDArray[floating]]] | NDArray[floating]
)
type IntArray = Sequence[int] | NDArray[integer]
