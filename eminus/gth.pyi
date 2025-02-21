# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import complexfloating, floating, integer
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

type _Int = integer[Any]
type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ArrayComplex = NDArray[_Complex]
type _ArrayInt = NDArray[_Int]

class GTH:
    GTH: dict[str, dict[str, float | _ArrayReal]]
    NbetaNL: int
    prj2beta: _ArrayInt
    betaNL: _ArrayComplex  # noqa: N815
    def __init__(self, scf: SCF | None = ...) -> None: ...
    def __getitem__(self, key: str) -> dict[str, float | _ArrayReal]: ...

def init_gth_loc(
    scf: SCF,
    **kwargs: Any,
) -> _ArrayComplex: ...
def init_gth_nonloc(
    atoms: Atoms,
    gth: GTH,
) -> tuple[int, _ArrayInt, _ArrayComplex]: ...
def calc_Vnonloc(
    scf: SCF,
    ik: int,
    spin: int,
    W: _ArrayComplex,
) -> _ArrayComplex: ...
def eval_proj_G(
    psp: dict[str, float | _ArrayReal],
    l: int,
    iprj: int,
    Gm: _ArrayReal,
    Omega: float,
) -> _ArrayReal: ...
