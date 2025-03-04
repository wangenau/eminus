# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .energies import Energy
from .gth import GTH
from .kpoints import KPoints
from .utils import BaseObject

type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ArrayComplex = NDArray[_Complex]

class SCF(BaseObject):
    etol: float
    gradtol: float | None
    sic: bool
    disp: bool | dict[str, bool | str | None]
    smear_update: int
    energies: Energy
    is_converged: bool
    gth: GTH
    Vloc: _ArrayComplex
    W: list[_ArrayComplex] | None
    Y: list[_ArrayComplex] | None
    Z: list[_ArrayComplex] | None
    D: list[_ArrayComplex] | None
    n: _ArrayReal | None
    n_spin: _ArrayReal | None
    dn_spin: _ArrayReal | None
    tau: _ArrayReal | None
    phi: _ArrayReal | None
    exc: _ArrayReal | None
    vxc: _ArrayComplex | None
    vsigma: _ArrayComplex | None
    vtau: _ArrayComplex | None
    def __init__(
        self,
        atoms: Atoms,
        xc: str = ...,
        pot: str = ...,
        guess: str = ...,
        etol: float = ...,
        gradtol: float | None = ...,
        sic: bool = ...,
        disp: bool | dict[str, bool | str | None] = ...,
        opt: dict[str, int] | None = ...,
        verbose: int | str | None = ...,
    ) -> None: ...
    @property
    def atoms(self) -> Atoms: ...
    @atoms.setter
    def atoms(self, value: Atoms) -> None: ...
    @property
    def xc(self) -> list[str]: ...
    @xc.setter
    def xc(self, value: str | Sequence[str]) -> None: ...
    @property
    def xc_params(self) -> dict[str, float]: ...
    @xc_params.setter
    def xc_params(self, value: dict[str, float]) -> None: ...
    @property
    def pot(self) -> str: ...
    @pot.setter
    def pot(self, value: str) -> None: ...
    @property
    def pot_params(self) -> dict[str, float]: ...
    @pot_params.setter
    def pot_params(self, value: dict[str, float]) -> None: ...
    @property
    def guess(self) -> str: ...
    @guess.setter
    def guess(self, value: str) -> None: ...
    @property
    def opt(self) -> dict[str, int]: ...
    @opt.setter
    def opt(self, value: dict[str, int]) -> None: ...
    @property
    def verbose(self) -> str: ...
    @verbose.setter
    def verbose(self, value: int | str | None) -> None: ...
    @property
    def kpts(self) -> KPoints: ...
    @property
    def pot_params_defaults(self) -> dict[str, float]: ...
    @property
    def psp(self) -> str: ...
    @property
    def symmetric(self) -> bool: ...
    @property
    def xc_type(self) -> str: ...
    @property
    def xc_params_defaults(self) -> dict[str, float]: ...
    def run(self, **kwargs: Any) -> float: ...
    kernel = run
    def converge_bands(self, **kwargs: Any) -> SCF: ...
    def converge_empty_bands(
        self,
        Nempty: int | None = ...,
        **kwargs: Any,
    ) -> SCF: ...
    def recenter(self, center: float | Sequence[float] | _ArrayReal | None = ...) -> SCF: ...
    def clear(self) -> SCF: ...
    @staticmethod
    def callback(scf: SCF, step: int) -> None: ...

class RSCF(SCF):
    @property
    def atoms(self) -> Atoms: ...
    @atoms.setter
    def atoms(self, value: Atoms) -> None: ...

class USCF(SCF):
    @property
    def atoms(self) -> Atoms: ...
    @atoms.setter
    def atoms(self, value: Atoms) -> None: ...
