# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from numpy import complex128, float64, int64
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

class GTH:
    GTH: dict[str, dict[str, float | NDArray[float64]]]
    NbetaNL: int
    prj2beta: NDArray[int64]
    betaNL: NDArray[complex128]  # noqa: N815
    def __init__(self, scf: SCF | None = ...) -> None: ...
    def __getitem__(self, key: str) -> dict[str, float | NDArray[float64]]: ...

def init_gth_loc(scf: SCF) -> NDArray[complex128]: ...
def init_gth_nonloc(
    atoms: Atoms,
    gth: GTH,
) -> tuple[int, NDArray[int64], NDArray[complex128]]: ...
def calc_Vnonloc(
    scf: SCF,
    ik: int,
    spin: int,
    W: NDArray[complex128],
) -> NDArray[complex128]: ...
def eval_proj_G(
    psp: dict[str, float | NDArray[float64]],
    l: int,
    iprj: int,
    Gm: NDArray[float64],
    Omega: float,
) -> NDArray[float64]: ...
