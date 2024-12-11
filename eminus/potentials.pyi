# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

from numpy import complexfloating
from numpy.typing import NDArray

from .scf import SCF

def harmonic(scf: SCF) -> NDArray[complexfloating]: ...
def coulomb(scf: SCF) -> NDArray[complexfloating]: ...
def ge(scf: SCF) -> NDArray[complexfloating]: ...
def init_pot(scf: SCF) -> NDArray[complexfloating]: ...

IMPLEMENTED: dict[str, Callable[[SCF], NDArray[complexfloating]]]
