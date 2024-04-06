# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

from numpy import complex128
from numpy.typing import NDArray

from .scf import SCF

def harmonic(scf: SCF) -> NDArray[complex128]: ...
def coulomb(scf: SCF) -> NDArray[complex128]: ...
def ge(scf: SCF) -> NDArray[complex128]: ...
def init_pot(scf: SCF) -> NDArray[complex128]: ...

IMPLEMENTED: dict[str, Callable[[SCF], NDArray[complex128]]]
