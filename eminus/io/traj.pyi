# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def read_traj(filename: str) -> list[tuple[list[str], NDArray[np.float64]]]: ...
def write_traj(
    obj: Atoms | SCF,
    filename: str,
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]] | None = ...,
    elec_symbols: Sequence[str] = ...,
) -> None: ...
