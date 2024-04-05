# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF
from .typing import Array1D, Array2D

def domain_cuboid(
    obj: Atoms | SCF,
    length: float | Array1D,
    centers: Array1D | Array2D | None = ...,
) -> NDArray[np.bool_]: ...
def domain_isovalue(
    field: NDArray[np.float64],
    isovalue: float,
) -> NDArray[np.bool_]: ...
def domain_sphere(
    obj: Atoms | SCF,
    radius: float,
    centers: Array1D | Array2D | None = ...,
) -> NDArray[np.bool_]: ...
def truncate(
    field: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.bool_]: ...
