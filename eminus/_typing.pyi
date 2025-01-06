# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from numpy import floating, integer
from numpy.typing import NDArray

type _Array1D = Sequence[float] | NDArray[floating]  # noqa: PYI047
type _Array2D = Sequence[Sequence[float]] | Sequence[NDArray[floating]] | NDArray[floating]  # noqa: PYI047
type _Array3D = (  # noqa: PYI047
    Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[NDArray[floating]]] | NDArray[floating]
)
type _IntArray = Sequence[int] | NDArray[integer]  # noqa: PYI047
