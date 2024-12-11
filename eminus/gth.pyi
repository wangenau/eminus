# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from numpy import complexfloating, floating, integer
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

class GTH:
    GTH: dict[str, dict[str, float | NDArray[floating]]]
    NbetaNL: int
    prj2beta: NDArray[integer]
    betaNL: NDArray[complexfloating]  # noqa: N815
    def __init__(self, scf: SCF | None = ...) -> None: ...
    def __getitem__(self, key: str) -> dict[str, float | NDArray[floating]]: ...

def init_gth_loc(scf: SCF) -> NDArray[complexfloating]: ...
def init_gth_nonloc(
    atoms: Atoms,
    gth: GTH,
) -> tuple[int, NDArray[integer], NDArray[complexfloating]]: ...
def calc_Vnonloc(
    scf: SCF,
    ik: int,
    spin: int,
    W: NDArray[complexfloating],
) -> NDArray[complexfloating]: ...
def eval_proj_G(
    psp: dict[str, float | NDArray[floating]],
    l: int,
    iprj: int,
    Gm: NDArray[floating],
    Omega: float,
) -> NDArray[floating]: ...
