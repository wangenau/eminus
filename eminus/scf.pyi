# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import complexfloating, floating
from numpy.typing import NDArray

from ._typing import _Array1D
from .atoms import Atoms
from .energies import Energy
from .gth import GTH
from .kpoints import KPoints
from .utils import BaseObject

class SCF(BaseObject):
    etol: float
    gradtol: float | None
    sic: bool
    disp: bool | dict[str, bool | str | None]
    smear_update: int
    energies: Energy
    is_converged: bool
    gth: GTH
    Vloc: NDArray[complexfloating]
    W: list[NDArray[complexfloating]] | None
    Y: list[NDArray[complexfloating]] | None
    Z: list[NDArray[complexfloating]] | None
    D: list[NDArray[complexfloating]] | None
    n: NDArray[floating] | None
    n_spin: NDArray[floating] | None
    dn_spin: NDArray[floating] | None
    tau: NDArray[floating] | None
    phi: NDArray[floating] | None
    exc: NDArray[floating] | None
    vxc: NDArray[complexfloating] | None
    vsigma: NDArray[complexfloating] | None
    vtau: NDArray[complexfloating] | None
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
    def xc_params(self) -> dict[str, Any]: ...
    @xc_params.setter
    def xc_params(self, value: dict[str, Any]) -> None: ...
    @property
    def pot(self) -> str: ...
    @pot.setter
    def pot(self, value: str) -> None: ...
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
    def psp(self) -> str: ...
    @property
    def symmetric(self) -> bool: ...
    @property
    def xc_type(self) -> str: ...
    @property
    def xc_params_defaults(self) -> dict[str, Any]: ...
    def run(self, **kwargs: Any) -> float: ...
    kernel = run
    def converge_bands(self, **kwargs: Any) -> SCF: ...
    def converge_empty_bands(
        self,
        Nempty: int | None = ...,
        **kwargs: Any,
    ) -> SCF: ...
    def recenter(self, center: float | _Array1D | None = ...) -> SCF: ...
    def clear(self) -> SCF: ...

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
