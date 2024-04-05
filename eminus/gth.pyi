# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

class GTH:
    GTH: dict[str, dict[str, int | float | NDArray[np.float64]]]
    NbetaNL: int
    prj2beta: NDArray[np.int64]
    betaNL: NDArray[np.complex128]
    def __init__(self, scf: SCF | None = ...) -> None: ...
    def __getitem__(self, key: str) -> dict[str, int | float | NDArray[np.float64]]: ...

def init_gth_loc(scf: SCF) -> NDArray[np.complex128]: ...
def init_gth_nonloc(
    atoms: Atoms,
    gth: GTH,
) -> tuple[int, NDArray[np.int64], NDArray[np.complex128]]: ...
def calc_Vnonloc(
    scf: SCF,
    ik: int,
    spin: int,
    W: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
def eval_proj_G(
    psp: dict[str, int | float | NDArray[np.float64]],
    l: int,
    iprj: int,
    Gm: NDArray[np.float64],
    Omega: float,
) -> NDArray[np.float64]: ...
